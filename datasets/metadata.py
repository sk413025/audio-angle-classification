"""
Dataset Metadata Management

This module provides classes for managing dataset metadata,
including sample information and dataset versioning.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import config  # 導入配置文件

@dataclass
class SampleMetadata:
    """
    Metadata for an audio sample.
    
    Attributes:
        id: Unique sample identifier
        file_path: Path to the audio file
        angle: Angle value (in degrees)
        material: Material type
        frequency: Frequency value
        seq_num: Sequence number
        excluded: Whether this sample is excluded
        notes: Additional notes or reasons for exclusion
        ghm_bins: History of GHM bin assignments (epoch -> bin)
    """
    id: str
    file_path: str
    angle: float
    material: str
    frequency: str
    seq_num: int
    excluded: bool = False
    notes: str = ""
    ghm_bins: Dict[int, int] = None  # epoch -> bin mapping
    
    def __post_init__(self):
        if self.ghm_bins is None:
            self.ghm_bins = {}
            
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SampleMetadata':
        """Create from dictionary"""
        return cls(**data)
    
    def add_ghm_bin(self, epoch: int, bin_idx: int) -> None:
        """Record GHM bin assignment for an epoch"""
        self.ghm_bins[str(epoch)] = bin_idx
        
    def get_ghm_bin_history(self) -> Dict[str, int]:
        """Get history of GHM bin assignments"""
        return self.ghm_bins


class DatasetConfig:
    """
    Configuration management for datasets.
    
    Handles:
    - Dataset versioning
    - Exclusion lists
    - Configuration serialization
    """
    
    def __init__(self, 
                 version: str = "1.0", 
                 exclusions: Optional[str] = None,
                 metadata_file: Optional[str] = None,
                 description: str = ""):
        """
        Initialize dataset configuration.
        
        Args:
            version: Dataset version string
            exclusions: Path to exclusion list file (or None)
            metadata_file: Path to metadata JSON file (or None)
            description: Human-readable description
        """
        self.version = version
        self.exclusion_file = exclusions
        self.metadata_file = metadata_file
        self.description = description
        
    def get_exclusion_path(self) -> str:
        """
        Get absolute path to exclusion file.
        
        Returns:
            Path to exclusion file for this dataset version
        """
        if self.exclusion_file:
            return self.exclusion_file
            
        # Default path based on version
        return os.path.join(
            config.DATA_ROOT, 
            'exclusions', 
            f'exclusions_v{self.version}.txt'
        )
    
    def get_metadata_path(self) -> str:
        """
        Get absolute path to metadata file.
        
        Returns:
            Path to metadata file for this dataset version
        """
        if self.metadata_file:
            return self.metadata_file
            
        # Default path based on version
        return os.path.join(
            config.DATA_ROOT, 
            'metadata',
            f'metadata_v{self.version}.json'
        )
    
    def save(self, path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            path: Path to save configuration
        """
        data = {
            "version": self.version,
            "exclusion_file": self.exclusion_file,
            "metadata_file": self.metadata_file,
            "description": self.description
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DatasetConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            DatasetConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
            
        return cls(
            version=data.get("version", "1.0"),
            exclusions=data.get("exclusion_file"),
            metadata_file=data.get("metadata_file"),
            description=data.get("description", "")
        )


def analyze_ghm_samples(metadata_file: str, bin_idx: int, epoch: Optional[int] = None) -> List[str]:
    """
    Analyze samples in a specific GHM bin.
    
    Args:
        metadata_file: Path to metadata file
        bin_idx: GHM bin index to analyze
        epoch: Specific epoch to analyze (None for latest)
        
    Returns:
        List of sample IDs in the specified bin
    """
    result = []
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        for sample_id, sample_data in metadata.items():
            ghm_bins = sample_data.get('ghm_bins', {})
            
            if not ghm_bins:
                continue
                
            # If epoch specified, check that epoch
            if epoch is not None and str(epoch) in ghm_bins:
                if ghm_bins[str(epoch)] == bin_idx:
                    result.append(sample_id)
            # Otherwise, check the latest epoch
            else:
                # Convert to integers for proper sorting
                epochs = [int(e) for e in ghm_bins.keys()]
                if epochs:
                    latest = max(epochs)
                    if ghm_bins[str(latest)] == bin_idx:
                        result.append(sample_id)
    
    except Exception as e:
        print(f"Error analyzing GHM samples: {e}")
    
    return result 