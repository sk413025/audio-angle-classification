"""
Base Dataset Classes

This module provides base classes for dataset management with
filtering capabilities and sample tracking.
"""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union

class ManagedDataset(Dataset):
    """
    Base dataset class with sample management features.
    
    Provides:
    - Sample filtering based on exclusion lists
    - Unique sample ID generation and tracking
    - Metadata association with samples
    """
    
    def __init__(self, 
                 data_root: str, 
                 exclusion_file: Optional[str] = None,
                 metadata_file: Optional[str] = None):
        """
        Initialize the managed dataset.
        
        Args:
            data_root: Root directory containing data files
            exclusion_file: Path to file listing excluded sample IDs (optional)
            metadata_file: Path to file with sample metadata (optional)
        """
        self.data_root = data_root
        self.samples = []  # Will be filled by child classes
        self.excluded_samples: Set[str] = set()
        self.sample_metadata: Dict[str, Dict] = {}
        
        # Load exclusion list if provided
        if exclusion_file and os.path.exists(exclusion_file):
            self._load_exclusion_list(exclusion_file)
            
        # Load metadata if provided
        if metadata_file and os.path.exists(metadata_file):
            self._load_metadata(metadata_file)
            
    def _load_exclusion_list(self, file_path: str) -> None:
        """
        Load the list of excluded sample IDs.
        
        Args:
            file_path: Path to the exclusion list file
        """
        with open(file_path, 'r') as f:
            for line in f:
                sample_id = line.strip()
                if sample_id and not sample_id.startswith('#'):  # Skip comments
                    self.excluded_samples.add(sample_id)
        print(f"Loaded {len(self.excluded_samples)} excluded samples")
    
    def _load_metadata(self, file_path: str) -> None:
        """
        Load sample metadata from JSON file.
        
        Args:
            file_path: Path to the metadata JSON file
        """
        try:
            with open(file_path, 'r') as f:
                self.sample_metadata = json.load(f)
            print(f"Loaded metadata for {len(self.sample_metadata)} samples")
        except Exception as e:
            print(f"Error loading metadata file {file_path}: {e}")
            self.sample_metadata = {}
    
    def exclude_sample(self, sample_id: str) -> None:
        """
        Mark a sample to be excluded from the dataset.
        
        Args:
            sample_id: Unique identifier for the sample
        """
        self.excluded_samples.add(sample_id)
    
    def include_sample(self, sample_id: str) -> None:
        """
        Remove a sample from the exclusion list.
        
        Args:
            sample_id: Unique identifier for the sample
        """
        if sample_id in self.excluded_samples:
            self.excluded_samples.remove(sample_id)
    
    def save_exclusion_list(self, output_file: str) -> None:
        """
        Save the current exclusion list to a file.
        
        Args:
            output_file: Path to save the exclusion list
        """
        with open(output_file, 'w') as f:
            f.write("# Sample exclusion list - one sample ID per line\n")
            f.write("# Generated on: " + torch.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            for sample_id in sorted(self.excluded_samples):
                f.write(f"{sample_id}\n")
        print(f"Saved {len(self.excluded_samples)} excluded samples to {output_file}")
    
    def save_metadata(self, output_file: str) -> None:
        """
        Save the current sample metadata to a JSON file.
        
        Args:
            output_file: Path to save the metadata
        """
        with open(output_file, 'w') as f:
            json.dump(self.sample_metadata, f, indent=2)
        print(f"Saved metadata for {len(self.sample_metadata)} samples to {output_file}")
    
    def get_sample_metadata(self, sample_id: str) -> Dict:
        """
        Get metadata for a specific sample.
        
        Args:
            sample_id: Unique identifier for the sample
            
        Returns:
            Dictionary of metadata or empty dict if not found
        """
        return self.sample_metadata.get(sample_id, {})
    
    def set_sample_metadata(self, sample_id: str, metadata: Dict) -> None:
        """
        Set metadata for a specific sample.
        
        Args:
            sample_id: Unique identifier for the sample
            metadata: Dictionary of metadata to store
        """
        self.sample_metadata[sample_id] = metadata
    
    def get_valid_indices(self) -> List[int]:
        """
        Get indices of all valid (non-excluded) samples.
        
        Returns:
            List of valid sample indices
        """
        return [i for i, sample in enumerate(self.samples) 
                if self.get_sample_id(sample) not in self.excluded_samples]
    
    def get_sample_id(self, sample: Any) -> str:
        """
        Generate a unique ID for a sample.
        Must be implemented by child classes.
        
        Args:
            sample: Sample data
            
        Returns:
            Unique string identifier for the sample
        """
        raise NotImplementedError("Child classes must implement get_sample_id")
    
    def _get_sample(self, idx: int) -> Any:
        """
        Get a sample by its index. 
        Must be implemented by child classes.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample data
        """
        raise NotImplementedError("Child classes must implement _get_sample")
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample by index, incorporating filtering logic.
        
        Args:
            idx: Index in the filtered dataset
            
        Returns:
            Sample data with metadata
        """
        valid_indices = self.get_valid_indices()
        if idx >= len(valid_indices):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(valid_indices)} valid samples")
        
        real_idx = valid_indices[idx]
        sample_data = self._get_sample(real_idx)
        
        # Generate sample ID for tracking
        sample = self.samples[real_idx]
        sample_id = self.get_sample_id(sample)
        
        # Return the sample data along with its ID for tracking
        if isinstance(sample_data, tuple):
            return (*sample_data, sample_id)
        else:
            return (sample_data, sample_id)
    
    def __len__(self) -> int:
        """
        Get the number of valid (non-excluded) samples.
        
        Returns:
            Number of valid samples
        """
        return len(self.get_valid_indices()) 