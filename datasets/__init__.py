"""
Dataset Management Module

This module handles audio data loading, preprocessing, and management.
It provides facilities for:
- Loading and processing audio data
- Creating spectrograms
- Managing sample metadata
- Filtering and excluding problematic samples
- Tracking samples during training

Main components:
- base.py: Base dataset classes with common functionality
- audio_dataset.py: Audio and spectrogram dataset implementations
- ranking_dataset.py: Dataset for ranking pairs
- metadata.py: Sample metadata management
"""

from datasets.base import ManagedDataset
from datasets.audio_dataset import (
    AudioSpectrumDataset, 
    SpectrogramDatasetWithMaterial  # 向後兼容
)
from datasets.ranking_dataset import RankingPairDataset, GHMAwareRankingDataset
from datasets.metadata import DatasetConfig, SampleMetadata

# Import legacy class for backward compatibility
# (This allows existing code to continue working with minimal changes)
from datasets.audio_dataset import SpectrogramDatasetWithMaterial as LegacySpectrogramDatasetWithMaterial 