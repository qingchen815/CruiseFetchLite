import numpy as np

class DPFMetadataManager:
    '''
    Manager for Decoupled Positional Frequency (DPF) metadata
    '''
    
    def __init__(self, num_candidates=4):
        '''
        Initialize DPF metadata manager
        
        Args:
            num_candidates: Number of candidates to track for each page
        '''
        self.num_candidates = num_candidates
        self.page_metadata = {}  # Maps page_id -> DPF metadata
        self.page_cluster_map = {}  # Maps page_id -> cluster_id
        
        # Aggregate offset transitions per cluster
        self.cluster_offset_transitions = []
    
    def update_page_access(self, trigger_page, next_page, trigger_offset, next_offset):
        '''
        Update DPF metadata for a page access
        
        Args:
            trigger_page: Page ID of the trigger access
            next_page: Page ID of the next access
            trigger_offset: Offset of the trigger access
            next_offset: Offset of the next access
        '''
        # Initialize metadata for trigger page if not exists
        if trigger_page not in self.page_metadata:
            self.page_metadata[trigger_page] = {
                'successors': {},  # Maps successor_page -> frequency
                'offset_transitions': np.zeros((64, 64), dtype=np.int32)  # [trigger_offset, next_offset]
            }
        
        # Update successor frequency
        successors = self.page_metadata[trigger_page]['successors']
        if next_page in successors:
            successors[next_page] += 1
        else:
            successors[next_page] = 1
        
        # Update offset transitions
        self.page_metadata[trigger_page]['offset_transitions'][trigger_offset, next_offset] += 1
    
    def get_candidate_pages(self, trigger_page):
        '''
        Get the top N candidate pages for a trigger page
        
        Args:
            trigger_page: Trigger page ID
            
        Returns:
            List of (page_id, frequency) pairs sorted by frequency
        '''
        if trigger_page not in self.page_metadata:
            return []
        
        # Get successors and sort by frequency
        successors = self.page_metadata[trigger_page]['successors']
        sorted_successors = sorted(successors.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N candidates
        return sorted_successors[:self.num_candidates]
    
    def get_dpf_vector(self, trigger_page):
        '''
        Get the DPF vector for a trigger page
        
        Args:
            trigger_page: Trigger page ID
            
        Returns:
            DPF vector as numpy array
        '''
        candidates = self.get_candidate_pages(trigger_page)
        
        # Create DPF vector
        dpf_vector = np.zeros(self.num_candidates, dtype=np.float32)
        
        # Fill with frequencies
        total_freq = sum(freq for _, freq in candidates)
        if total_freq > 0:
            for i, (_, freq) in enumerate(candidates):
                if i < self.num_candidates:
                    dpf_vector[i] = freq / total_freq
        
        return dpf_vector
    
    def update_cluster_mapping(self, page_id, cluster_mapping_utils):
        '''
        Update the cluster mapping for a page based on its offset transitions
        
        Args:
            page_id: Page ID to update
            cluster_mapping_utils: Utility for cluster mapping
            
        Returns:
            Assigned cluster ID
        '''
        # Skip if page has no metadata yet
        if page_id not in self.page_metadata:
            return 0
        
        # Get offset transitions for this page
        offset_transitions = self.page_metadata[page_id]['offset_transitions']
        
        # Determine cluster using dynamic mapping
        cluster_id = cluster_mapping_utils.dynamic_page_to_cluster_mapping(
            offset_transitions, 
            self.cluster_offset_transitions
        )
        
        # Update mapping
        self.page_cluster_map[page_id] = cluster_id
        
        return cluster_id
    
    def get_cluster_id(self, page_id, cluster_mapping_utils):
        '''
        Get the cluster ID for a page
        
        Args:
            page_id: Page ID
            cluster_mapping_utils: Utility for cluster mapping
            
        Returns:
            Cluster ID
        '''
        # If page not mapped yet, update mapping
        if page_id not in self.page_cluster_map:
            return self.update_cluster_mapping(page_id, cluster_mapping_utils)
        
        return self.page_cluster_map[page_id]
    
    def prepare_model_inputs(self, page_history, offset_history, load_pc, cluster_mapping_utils):
        '''

        Prepare inputs for the T-LITE model
        
        Args:
            page_history: History of page IDs
            offset_history: History of offsets
            load_pc: Load PC
            cluster_mapping_utils: Utility for cluster mapping
            
        Returns:
            Tuple of (cluster_history, offset_history, pc, dpf_vectors)
        '''
        # Map pages to clusters
        cluster_history = [self.get_cluster_id(page, cluster_mapping_utils) for page in page_history]
        
        # Get DPF vector for the most recent page
        trigger_page = page_history[-1]
        dpf_vector = self.get_dpf_vector(trigger_page)
        
        # Reshape inputs for the model
        cluster_history = np.array(cluster_history, dtype=np.int32)
        dpf_vectors = np.array([dpf_vector], dtype=np.float32)  # [1, num_candidates]
        
        return cluster_history, offset_history, load_pc, dpf_vectors
    
    def get_metadata_size_kb(self):
        '''
        Calculate the size of the metadata in KB
        
        Returns:
            Metadata size in KB
        '''
        total_bytes = 0
        
        # Size of page_metadata
        for page_id, metadata in self.page_metadata.items():
            # successors dict: ~12 bytes per entry (key, value, overhead)
            successors_size = len(metadata['successors']) * 12
            
            # offset_transitions: 64x64 matrix of int32 (4 bytes each)
            offset_transitions_size = 64 * 64 * 4
            
            total_bytes += successors_size + offset_transitions_size
        
        # Size of page_cluster_map: ~12 bytes per entry
        total_bytes += len(self.page_cluster_map) * 12
        
        # Convert to KB
        return total_bytes / 1024
    
    def serialize(self, filepath):
        '''
        Serialize metadata to a file
        
        Args:
            filepath: Path to save serialized metadata
        '''
        metadata = {
            'num_candidates': self.num_candidates,
            'page_metadata': self.page_metadata,
            'page_cluster_map': self.page_cluster_map,
            'cluster_offset_transitions': self.cluster_offset_transitions
        }
        
        np.save(filepath, metadata, allow_pickle=True)
    
    @classmethod
    def deserialize(cls, filepath):
        '''
        Deserialize metadata from a file
        
        Args:
            filepath: Path to serialized metadata
            
        Returns:
            Deserialized DPFMetadataManager
        '''
        metadata = np.load(filepath, allow_pickle=True).item()
        
        manager = cls(num_candidates=metadata['num_candidates'])
        manager.page_metadata = metadata['page_metadata']
        manager.page_cluster_map = metadata['page_cluster_map']
        manager.cluster_offset_transitions = metadata['cluster_offset_transitions']
        
        return manager


class DPFMetadataCache:
    '''
    Cache for DPF metadata to simulate on-chip storage constraints
    '''
    
    def __init__(self, size_kb=64, num_ways=8):
        '''
        Initialize DPF metadata cache
        
        Args:
            size_kb: Cache size in KB
            num_ways: Cache associativity
        '''
        self.size_kb = size_kb
        self.num_ways = num_ways
        
        # Calculate number of sets based on size and associativity
        # Each entry is approximately 16KB (64x64x4 bytes for offset transitions + overhead)
        self.entries_capacity = int((size_kb * 1024) / 16384)
        self.num_sets = max(1, self.entries_capacity // num_ways)
        
        # Initialize cache
        self.cache = [{} for _ in range(self.num_sets)]  # List of sets, each set is a dict of {tag: {data, lru_counter}}
        self.lru_counter = 0
    
    def _get_set_index(self, page_id):
        '''Get the set index for a page ID'''
        return page_id % self.num_sets
    
    def _get_tag(self, page_id):
        '''Get the tag for a page ID'''
        return page_id // self.num_sets
    
    def lookup(self, page_id):
        '''
        Lookup metadata for a page
        
        Args:
            page_id: Page ID
            
        Returns:
            Metadata if found, None otherwise
        '''
        set_index = self._get_set_index(page_id)
        tag = self._get_tag(page_id)
        
        # Check if tag is in the set
        if tag in self.cache[set_index]:
            # Update LRU counter
            self.lru_counter += 1
            self.cache[set_index][tag]['lru_counter'] = self.lru_counter
            
            # Return metadata
            return self.cache[set_index][tag]['data']
        
        return None
    
    def insert(self, page_id, metadata):
        '''
        Insert metadata for a page
        
        Args:
            page_id: Page ID
            metadata: Metadata to insert
        '''
        set_index = self._get_set_index(page_id)
        tag = self._get_tag(page_id)
        
        # Check if tag is already in the set
        if tag in self.cache[set_index]:
            # Update metadata and LRU counter
            self.lru_counter += 1
            self.cache[set_index][tag]['data'] = metadata
            self.cache[set_index][tag]['lru_counter'] = self.lru_counter
            return
        
        # If set is full, evict LRU entry
        if len(self.cache[set_index]) >= self.num_ways:
            lru_tag = min(self.cache[set_index].items(), key=lambda x: x[1]['lru_counter'])[0]
            del self.cache[set_index][lru_tag]
        
        # Insert new entry
        self.lru_counter += 1
        self.cache[set_index][tag] = {
            'data': metadata,
            'lru_counter': self.lru_counter
        }
    
    def get_stats(self):
        '''
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        '''
        # Count number of entries
        entries = sum(len(s) for s in self.cache)
        
        return {
            'size_kb': self.size_kb,
            'num_ways': self.num_ways,
            'num_sets': self.num_sets,
            'entries_capacity': self.entries_capacity,
            'entries_used': entries,
            'utilization': entries / self.entries_capacity
        }
