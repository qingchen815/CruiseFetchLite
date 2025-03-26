"""
TLITE2: A lightweight neural prefetcher with behavioral clustering
"""

from model import create_tlite_model
from config import TLITEConfig, load_config, extend_voyager_config_for_tlite
from metrics import PrefetchMetrics
from prefetcher import TLITEPrefetcher
from metadata import DPFMetadataManager
from clustering import BehavioralClusteringUtils

__version__ = '1.0.0'

# Export main components
__all__ = [
    'create_tlite_model',
    'TLITEConfig',
    'load_config',
    'extend_voyager_config_for_tlite',
    'PrefetchMetrics',
    'TLITEPrefetcher',
    'DPFMetadataManager',
    'BehavioralClusteringUtils'
]
