import numpy as np
from tlite.metadata import DPFMetadataManager
from tlite.clustering import BehavioralClusteringUtils

class TLITEPrefetcher:
    '''
    T-LITE Prefetcher implementation
    '''
    
    def __init__(self, model, config):
        '''
        Initialize the T-LITE prefetcher
        
        Args:
            model: Trained T-LITE model
            config: Configuration object
        '''
        self.model = model
        self.config = config
        self.metadata_manager = DPFMetadataManager(num_candidates=config.num_candidates)
        self.cluster_mapping_utils = BehavioralClusteringUtils()
        
        # History buffers
        self.page_history = []
        self.offset_history = []
        self.history_length = config.history_length
        
        # Initialize histories with zeros
        for _ in range(self.history_length):
            self.page_history.append(0)
            self.offset_history.append(0)
        
        # Performance tracking
        self.stats = {
            'accesses': 0,
            'prefetches_issued': 0,
            'prefetches_useful': 0,
            'prediction_latency_cycles': 0,
            'metadata_size_kb': 0,
            'no_prefetch_predictions': 0
        }
        
        # Future accesses for tracking coverage (sliding window)
        self.future_accesses = set()
        self.future_window = 100  # Number of future accesses to track
        
        # Speculative prediction state
        self.previous_prefetch = None
        self.prefetch_was_correct = False
    
    def update_history(self, page, offset):
        '''
        Update access history with a new access
        
        Args:
            page: Page ID
            offset: Offset
        '''
        # Remove oldest entry
        self.page_history.pop(0)
        self.offset_history.pop(0)
        
        # Add new entry
        self.page_history.append(page)
        self.offset_history.append(offset)
    
    def record_access(self, addr, pc):
        '''
        Record a memory access
        
        Args:
            addr: Memory address
            pc: Program counter
        '''
        # Extract page and offset
        page = addr >> 6  # Get cache line address
        page_id = page >> self.config.offset_bits  # Extract page
        offset = page & ((1 << self.config.offset_bits) - 1)  # Extract offset within page
        
        # Update performance tracking
        self.stats['accesses'] += 1
        
        # Check if this access was prefetched
        if addr == self.previous_prefetch:
            self.prefetch_was_correct = True
            self.stats['prefetches_useful'] += 1
        
        # Update future accesses for coverage tracking
        self.future_accesses.add(addr)
        if len(self.future_accesses) > self.future_window:
            # Remove older accesses (approximate)
            self.future_accesses = set(list(self.future_accesses)[-self.future_window:])
        
        # Update DPF metadata with previous access
        if len(self.page_history) > 0 and self.page_history[-1] != 0:
            prev_page = self.page_history[-1]
            prev_offset = self.offset_history[-1]
            self.metadata_manager.update_page_access(prev_page, page_id, prev_offset, offset)
        
        # Update history
        self.update_history(page_id, offset)
        
        # Update metadata size stats

        self.stats['metadata_size_kb'] = self.metadata_manager.get_metadata_size_kb()
    
    def generate_prefetch(self, pc):
        '''
        Generate a prefetch based on current state
        
        Args:
            pc: Program counter for the current access
            
        Returns:
            Prefetch address or None if no prefetch
        '''
        # Prepare model inputs
        cluster_history, offset_history, load_pc, dpf_vectors = self.metadata_manager.prepare_model_inputs(
            self.page_history, 
            self.offset_history, 
            pc, 
            self.cluster_mapping_utils
        )
        
        # Calculate prediction latency
        prediction_latency = 29  # Base latency in cycles
        
        # Speculative prediction - no latency if previous prefetch was correct
        if self.prefetch_was_correct:
            prediction_latency = 0
        
        self.stats['prediction_latency_cycles'] += prediction_latency
        
        # Reset speculative state
        self.prefetch_was_correct = False
        
        # Get model predictions
        candidate_logits, offset_logits = self.model((
            np.array([cluster_history]),
            np.array([offset_history]),
            np.array([[pc]]),
            np.array([dpf_vectors])
        ))
        
        # Convert logits to predictions
        candidate_pred = np.argmax(candidate_logits[0])
        offset_pred = np.argmax(offset_logits[0])
        
        # Check if "no prefetch" was predicted
        if candidate_pred == self.config.num_candidates:
            self.stats['no_prefetch_predictions'] += 1
            return None
        
        # Get the candidate page ID from the DPF metadata
        candidates = self.metadata_manager.get_candidate_pages(self.page_history[-1])
        if not candidates or candidate_pred >= len(candidates):
            return None
        
        prefetch_page = candidates[candidate_pred][0]
        
        # Compose the final prefetch address
        prefetch_addr = (prefetch_page << self.config.offset_bits) | offset_pred
        prefetch_addr = prefetch_addr << 6  # Convert to byte address
        
        # Update stats
        self.stats['prefetches_issued'] += 1
        
        # Store for speculative prediction
        self.previous_prefetch = prefetch_addr
        
        return prefetch_addr
    
    def handle_access(self, addr, pc):
        '''
        Handle a memory access and potentially return a prefetch
        
        Args:
            addr: Memory address
            pc: Program counter
            
        Returns:
            Prefetch address or None
        '''
        # First generate a prefetch based on current state
        prefetch = self.generate_prefetch(pc)
        
        # Then update internal state with this access
        self.record_access(addr, pc)
        
        return prefetch
    
    def get_accuracy(self):
        '''Get prefetch accuracy'''
        if self.stats['prefetches_issued'] == 0:
            return 0.0
        return (self.stats['prefetches_useful'] / self.stats['prefetches_issued']) * 100
    
    def get_coverage(self):
        '''Get prefetch coverage'''
        if self.stats['accesses'] == 0:
            return 0.0
        return (self.stats['prefetches_useful'] / self.stats['accesses']) * 100
    
    def print_stats(self):
        '''Print prefetcher statistics'''
        print("T-LITE Prefetcher Statistics:")
        print(f"  Accesses: {self.stats['accesses']}")
        print(f"  Prefetches Issued: {self.stats['prefetches_issued']}")
        print(f"  Useful Prefetches: {self.stats['prefetches_useful']}")
        print(f"  No-Prefetch Predictions: {self.stats['no_prefetch_predictions']}")
        print(f"  Accuracy: {self.get_accuracy():.2f}%")
        print(f"  Coverage: {self.get_coverage():.2f}%")
        print(f"  Metadata Size: {self.stats['metadata_size_kb']:.2f} KB")
        print(f"  Avg. Prediction Latency: {self.stats['prediction_latency_cycles'] / max(1, self.stats['accesses']):.2f} cycles")


class TwilightPrefetcher:
    '''
    Twilight Prefetcher implementation
    
    Similar to T-LITE but uses page embeddings instead of behavioral clusters
    '''
    
    def __init__(self, model, config):
        '''
        Initialize the Twilight prefetcher
        
        Args:
            model: Trained Twilight model
            config: Configuration object
        '''
        self.model = model
        self.config = config
        self.metadata_manager = DPFMetadataManager(num_candidates=config.num_candidates)
        
        # History buffers
        self.page_history = []
        self.offset_history = []
        self.history_length = config.history_length
        
        # Initialize histories with zeros
        for _ in range(self.history_length):
            self.page_history.append(0)
            self.offset_history.append(0)
        
        # Performance tracking
        self.stats = {
            'accesses': 0,
            'prefetches_issued': 0,
            'prefetches_useful': 0,
            'prediction_latency_cycles': 0,
            'metadata_size_kb': 0,
            'no_prefetch_predictions': 0
        }
        
        # Future accesses for tracking coverage (sliding window)
        self.future_accesses = set()
        self.future_window = 100  # Number of future accesses to track
        
        # Speculative prediction state
        self.previous_prefetch = None
        self.prefetch_was_correct = False
    
    def update_history(self, page, offset):
        '''Update access history with a new access'''
        # Remove oldest entry
        self.page_history.pop(0)
        self.offset_history.pop(0)
        
        # Add new entry
        self.page_history.append(page)
        self.offset_history.append(offset)
    
    def record_access(self, addr, pc):
        '''Record a memory access'''
        # Extract page and offset
        page = addr >> 6  # Get cache line address
        page_id = page >> self.config.offset_bits  # Extract page
        offset = page & ((1 << self.config.offset_bits) - 1)  # Extract offset within page
        
        # Update performance tracking
        self.stats['accesses'] += 1
        
        # Check if this access was prefetched
        if addr == self.previous_prefetch:
            self.prefetch_was_correct = True
            self.stats['prefetches_useful'] += 1
        
        # Update future accesses for coverage tracking
        self.future_accesses.add(addr)
        if len(self.future_accesses) > self.future_window:
            # Remove older accesses (approximate)
            self.future_accesses = set(list(self.future_accesses)[-self.future_window:])
        
        # Update DPF metadata with previous access
        if len(self.page_history) > 0 and self.page_history[-1] != 0:
            prev_page = self.page_history[-1]
            prev_offset = self.offset_history[-1]
            self.metadata_manager.update_page_access(prev_page, page_id, prev_offset, offset)
        
        # Update history
        self.update_history(page_id, offset)
        
        # Update metadata size stats
        self.stats['metadata_size_kb'] = self.metadata_manager.get_metadata_size_kb()
    
    def generate_prefetch(self, pc):
        '''Generate a prefetch based on current state'''
        # Twilight uses pages directly instead of clusters in inputs
        # DPF vectors are the same
        dpf_vector = self.metadata_manager.get_dpf_vector(self.page_history[-1])
        
        # Calculate prediction latency (slower than T-LITE due to larger model)
        prediction_latency = 82  # Base latency in cycles
        
        # Speculative prediction - no latency if previous prefetch was correct
        if self.prefetch_was_correct:
            prediction_latency = 0
        
        self.stats['prediction_latency_cycles'] += prediction_latency
        
        # Reset speculative state
        self.prefetch_was_correct = False
        
        # Get model predictions
        candidate_logits, offset_logits = self.model((
            np.array([self.page_history]),
            np.array([self.offset_history]),
            np.array([[pc]]),
            np.array([[dpf_vector]])
        ))
        
        # Convert logits to predictions
        candidate_pred = np.argmax(candidate_logits[0])
        offset_pred = np.argmax(offset_logits[0])
        
        # Check if "no prefetch" was predicted
        if candidate_pred == self.config.num_candidates:
            self.stats['no_prefetch_predictions'] += 1
            return None
        
        # Get the candidate page ID from the DPF metadata
        candidates = self.metadata_manager.get_candidate_pages(self.page_history[-1])
        if not candidates or candidate_pred >= len(candidates):
            return None
        
        prefetch_page = candidates[candidate_pred][0]
        
        # Compose the final prefetch address
        prefetch_addr = (prefetch_page << self.config.offset_bits) | offset_pred
        prefetch_addr = prefetch_addr << 6  # Convert to byte address
        
        # Update stats
        self.stats['prefetches_issued'] += 1
        
        # Store for speculative prediction
        self.previous_prefetch = prefetch_addr
        
        return prefetch_addr
    
    def handle_access(self, addr, pc):
        '''Handle a memory access and potentially return a prefetch'''
        # First generate a prefetch based on current state
        prefetch = self.generate_prefetch(pc)
        
        # Then update internal state with this access
        self.record_access(addr, pc)
        
        return prefetch
    
    def get_accuracy(self):
        '''Get prefetch accuracy'''
        if self.stats['prefetches_issued'] == 0:
            return 0.0
        return (self.stats['prefetches_useful'] / self.stats['prefetches_issued']) * 100
    
    def get_coverage(self):
        '''Get prefetch coverage'''
        if self.stats['accesses'] == 0:
            return 0.0
        return (self.stats['prefetches_useful'] / self.stats['accesses']) * 100
    
    def print_stats(self):
        '''Print prefetcher statistics'''
        print("Twilight Prefetcher Statistics:")
        print(f"  Accesses: {self.stats['accesses']}")
        print(f"  Prefetches Issued: {self.stats['prefetches_issued']}")
        print(f"  Useful Prefetches: {self.stats['prefetches_useful']}")
        print(f"  No-Prefetch Predictions: {self.stats['no_prefetch_predictions']}")
        print(f"  Accuracy: {self.get_accuracy():.2f}%")
        print(f"  Coverage: {self.get_coverage():.2f}%")
        print(f"  Metadata Size: {self.stats['metadata_size_kb']:.2f} KB")
        print(f"  Avg. Prediction Latency: {self.stats['prediction_latency_cycles'] / max(1, self.stats['accesses']):.2f} cycles")
