"""
Ranking Dataset Classes

This module provides dataset implementations for ranking tasks,
particularly for paired comparisons and margin ranking loss.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import pickle

class RankingPairDataset(Dataset):
    """
    Ranking pair dataset that generates pairs suitable for MarginRankingLoss.
    
    Returns (x1, x2, target, label1, label2, sample_id1, sample_id2) where:
    - target=1 means x1 should rank higher than x2 (x1 class > x2 class)
    - target=-1 means x2 should rank higher than x1 (x1 class < x2 class)
    """
    
    def __init__(self, dataset, pairs_file: Optional[str] = None, exclusions_file: Optional[str] = None):
        """
        Initialize ranking pair dataset.
        
        Args:
            dataset: Base dataset to draw samples from
            pairs_file: Optional path to saved pairs file
            exclusions_file: Optional path to a file listing excluded pair IDs
        """
        self.dataset = dataset
        self.pairs = []
        self.excluded_pairs: Set[str] = set()
        
        # 載入排除列表（如果提供）
        if exclusions_file and os.path.exists(exclusions_file):
            self._load_exclusions(exclusions_file)
            
        if pairs_file and os.path.exists(pairs_file):
            self._load_pairs(pairs_file)
        else:
            self.pairs = self._create_ranking_pairs()
            
        print(f"Created {len(self.pairs)} ranking pairs")
    
    def _load_exclusions(self, file_path: str) -> None:
        """
        Load the list of excluded pair IDs.
        
        Args:
            file_path: Path to the exclusion list file
        """
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments and empty lines
                    # 確保每行代表一個完整的配對ID
                    pair_id = line.strip()
                    
                    # 將配對ID添加到排除集合中
                    self.excluded_pairs.add(pair_id)
        
        print(f"Loaded {len(self.excluded_pairs)} excluded pairs from {file_path}")
    
    def _load_pairs(self, path: str) -> None:
        """
        Load predefined pairs from a file.
        
        Args:
            path: Path to pairs file (pickle format)
        """
        try:
            with open(path, 'rb') as f:
                loaded_pairs = pickle.load(f)
                
            # 過濾掉被排除的對
            self.pairs = []
            for pair in loaded_pairs:
                if not self._is_excluded_pair(pair):
                    self.pairs.append(pair)
                    
            print(f"Loaded {len(self.pairs)} valid pairs from {path} (excluded {len(loaded_pairs) - len(self.pairs)})")
        except Exception as e:
            print(f"Error loading pairs from {path}: {e}")
            self.pairs = self._create_ranking_pairs()
    
    def _is_excluded_pair(self, pair: Tuple[int, int, int]) -> bool:
        """
        Check if a pair should be excluded based on the exclusion list.
        
        Args:
            pair: The pair to check (idx1, idx2, target)
            
        Returns:
            True if the pair should be excluded, False otherwise
        """
        if not self.excluded_pairs:
            return False
            
        idx1, idx2, _ = pair
        
        # 獲取樣本ID
        sample1 = self.dataset[idx1]
        sample2 = self.dataset[idx2]
        
        # 處理不同的返回格式
        if len(sample1) >= 3:  # (data, label, sample_id, ...) format
            sample_id1 = sample1[2]
            sample_id2 = sample2[2]
        else:
            # 如果沒有樣本ID，使用索引作為回退
            return False
            
        # 構建排序對ID - 只檢查正向排序
        pair_id = f"{sample_id1}_{sample_id2}"
        
        # 只有當完全相同的排序對（包括順序）存在於排除列表中時才排除
        if pair_id in self.excluded_pairs:
            return True
            
        return False
    
    def save_pairs(self, path: str) -> None:
        """
        Save current pairs to a file.
        
        Args:
            path: Path to save pairs
        """
        with open(path, 'wb') as f:
            pickle.dump(self.pairs, f)
        print(f"Saved {len(self.pairs)} pairs to {path}")
    
    def _create_ranking_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Create ranking pairs based on class labels.
        
        Returns:
            List of tuple pairs (idx1, idx2, target)
        """
        pairs = []
        # 獲取所有樣本及標籤
        all_data = []
        for idx in range(len(self.dataset)):
            data = self.dataset[idx]
            if len(data) >= 2:  # Expecting at least (data, label)
                label = data[1]
                label = label.item() if isinstance(label, torch.Tensor) else label
                all_data.append((idx, label))
        
        # 確定生成多少對
        n_samples = len(all_data)
        # 這裡設置為原來可能對數的一半作為默認
        n_pairs = (n_samples * (n_samples - 1)) // 2
        
        # 隨機生成樣本對
        for _ in range(n_pairs):
            # 隨機選擇兩個不同的樣本
            sample1, sample2 = np.random.choice(len(all_data), 2, replace=False)
            idx1, label1 = all_data[sample1]
            idx2, label2 = all_data[sample2]
            
            # 根據類別順序確定目標值
            if label1 > label2:
                # x1 應排在 x2 前面
                pair = (idx1, idx2, 1)
                if not self._is_excluded_pair(pair):
                    pairs.append(pair)
            elif label1 < label2:
                # x2 應排在 x1 前面
                pair = (idx1, idx2, -1)
                if not self._is_excluded_pair(pair):
                    pairs.append(pair)
            # 如果標籤相同，則不添加此對（繼續循環）
        
        return pairs
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a ranking pair.
        
        Args:
            idx: Pair index
            
        Returns:
            Tuple of (data1, data2, target, label1, label2, sample_id1, sample_id2)
        """
        idx1, idx2, target = self.pairs[idx]
        
        # Get samples from underlying dataset
        sample1 = self.dataset[idx1]
        sample2 = self.dataset[idx2]
        
        # Handle different return formats based on underlying dataset
        if len(sample1) == 3:  # (data, label, sample_id) format
            data1, label1, sample_id1 = sample1
            data2, label2, sample_id2 = sample2
        elif len(sample1) == 2:  # (data, label) format
            data1, label1 = sample1
            data2, label2 = sample2
            sample_id1 = str(idx1)  # Use indices as fallback IDs
            sample_id2 = str(idx2)
        else:
            raise ValueError(f"Unexpected sample format: {sample1}")
        
        # Ensure target is a scalar float tensor
        target_tensor = torch.tensor(target, dtype=torch.float)
        
        return data1, data2, target_tensor, label1, label2, sample_id1, sample_id2
    
    def __len__(self) -> int:
        """
        Get number of ranking pairs.
        
        Returns:
            Number of pairs
        """
        return len(self.pairs)
    
    def get_pair_stats(self) -> Dict[str, int]:
        """
        Get statistics about the ranking pairs.
        
        Returns:
            Dictionary with pair statistics
        """
        targets = [t for _, _, t in self.pairs]
        positive = sum(1 for t in targets if t > 0)
        negative = sum(1 for t in targets if t < 0)
        
        return {
            "total_pairs": len(self.pairs),
            "positive_targets": positive,
            "negative_targets": negative,
            "target_balance": round(positive / len(self.pairs) if len(self.pairs) > 0 else 0, 2)
        }


class GHMAwareRankingDataset(RankingPairDataset):
    """
    Enhanced ranking dataset that handles GHM bin tracking.
    
    This dataset works with GHM loss functions to track which samples
    are assigned to which bins during training.
    """
    
    def __init__(self, dataset, pairs_file: Optional[str] = None, exclusions_file: Optional[str] = None):
        """
        Initialize GHM-aware ranking dataset.
        
        Args:
            dataset: Base dataset to draw samples from
            pairs_file: Optional path to saved pairs file
            exclusions_file: Optional path to a file listing excluded pair IDs
        """
        super().__init__(dataset, pairs_file, exclusions_file)
        self.ghm_bin_assignments = {}  # epoch -> {pair_idx -> bin_idx}
    
    def record_ghm_bin(self, idx: int, epoch: int, bin_idx: int) -> None:
        """
        Record GHM bin for a specific pair in a training epoch.
        
        Args:
            idx: Pair index
            epoch: Training epoch
            bin_idx: GHM bin index
        """
        if epoch not in self.ghm_bin_assignments:
            self.ghm_bin_assignments[epoch] = {}
        
        self.ghm_bin_assignments[epoch][idx] = bin_idx
    
    def get_pairs_in_bin(self, epoch: int, bin_idx: int) -> List[Tuple[int, int, int]]:
        """
        Get all pairs assigned to a specific bin in a given epoch.
        
        Args:
            epoch: Training epoch
            bin_idx: GHM bin index
            
        Returns:
            List of pairs (idx1, idx2, target) in the bin
        """
        if epoch not in self.ghm_bin_assignments:
            return []
        
        pairs_in_bin = []
        for idx, bin_assignment in self.ghm_bin_assignments[epoch].items():
            if bin_assignment == bin_idx and idx < len(self.pairs):
                pairs_in_bin.append(self.pairs[idx])
        
        return pairs_in_bin
    
    def save_bin_assignments(self, path: str) -> None:
        """
        Save bin assignments to a file.
        
        Args:
            path: Path to save assignments
        """
        with open(path, 'wb') as f:
            pickle.dump(self.ghm_bin_assignments, f)
    
    def load_bin_assignments(self, path: str) -> None:
        """
        Load bin assignments from a file.
        
        Args:
            path: Path to assignments file
        """
        with open(path, 'rb') as f:
            self.ghm_bin_assignments = pickle.load(f) 