"""
測試 TracIn 核心功能模組的單元測試

這些測試確保 TracIn 的核心功能正確工作，包括梯度計算、影響力計算等。
"""

import unittest
import torch
import os
import tempfile
import numpy as np
import json
from pathlib import Path

# 導入核心 TracIn 模組
from tracin.core.tracin import TracInCP, get_default_device
from tracin.core.ranking_tracin import RankingTracInCP


class SimpleModel(torch.nn.Module):
    """用於測試的簡單模型"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)


class SimpleRankingModel(torch.nn.Module):
    """用於測試排序任務的簡單模型"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)


class TestTracInCP(unittest.TestCase):
    """測試基礎 TracIn 實現"""
    
    def setUp(self):
        """設置測試環境"""
        self.device = torch.device('cpu')
        self.model = SimpleModel()
        
        # 創建臨時檢查點
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "model_checkpoint.pt")
        
        # 保存模型檢查點
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': 1
        }, self.checkpoint_path)
        
        # 初始化 TracIn
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.tracin = TracInCP(
            model=self.model,
            checkpoints=[self.checkpoint_path],
            loss_fn=self.loss_fn,
            device=self.device
        )
        
    def tearDown(self):
        """清理測試環境"""
        self.temp_dir.cleanup()
        
    def test_gradient_computation(self):
        """測試梯度計算功能"""
        # 創建測試數據
        inputs = torch.randn(2, 10)
        targets = torch.tensor([0, 1])
        
        # 計算梯度
        gradients = self.tracin.compute_gradients(inputs, targets, self.checkpoint_path)
        
        # 驗證
        self.assertIsInstance(gradients, list)
        self.assertTrue(len(gradients) > 0)
        for grad in gradients:
            self.assertIsInstance(grad, torch.Tensor)
            
    def test_influence_computation(self):
        """測試影響力計算功能"""
        # 創建簡單數據集
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = torch.randn(4, 10)
                self.targets = torch.tensor([0, 1, 0, 1])
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx], idx
        
        dataset = SimpleDataset()
        test_input = torch.randn(1, 10)
        test_target = torch.tensor([0])
        
        # 計算影響力
        influence_scores = self.tracin.compute_influence(
            dataset, test_input, test_target, batch_size=2, num_workers=0
        )
        
        # 驗證
        self.assertIsInstance(influence_scores, dict)
        self.assertEqual(len(influence_scores), 4)  # 4個訓練樣本
        
    def test_save_influence_scores(self):
        """測試保存影響力分數功能"""
        # 創建影響力分數
        influence_scores = {
            0: 0.1,
            1: -0.2,
            2: 0.3,
            3: -0.4
        }
        
        # 創建臨時文件
        metadata_path = os.path.join(self.temp_dir.name, "metadata.json")
        
        # 保存分數
        self.tracin.save_influence_scores(
            influence_scores, metadata_path, score_name="test_influence"
        )
        
        # 檢查文件是否存在
        self.assertTrue(os.path.exists(metadata_path))
        
        # 加載並驗證
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(len(metadata), 4)
        self.assertEqual(metadata["0"]["test_influence"], 0.1)
        self.assertEqual(metadata["1"]["test_influence"], -0.2)


class TestRankingTracInCP(unittest.TestCase):
    """測試排序任務的 TracIn 實現"""
    
    def setUp(self):
        """設置測試環境"""
        self.device = torch.device('cpu')
        self.model = SimpleRankingModel()
        
        # 創建臨時檢查點
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "model_checkpoint.pt")
        
        # 保存模型檢查點
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': 1
        }, self.checkpoint_path)
        
        # 初始化 RankingTracIn
        self.tracin = RankingTracInCP(
            model=self.model,
            checkpoints=[self.checkpoint_path],
            loss_type="standard",
            margin=1.0,
            device=self.device
        )
        
    def tearDown(self):
        """清理測試環境"""
        self.temp_dir.cleanup()
        
    def test_compute_gradients_for_pair(self):
        """測試對於排序對的梯度計算"""
        # 創建測試數據
        x1 = torch.randn(2, 10)
        x2 = torch.randn(2, 10)
        target = torch.tensor([1, -1])  # 1: x1 > x2, -1: x1 < x2
        
        # 計算梯度
        gradients = self.tracin.compute_gradients_for_pair(
            x1, x2, target, self.checkpoint_path
        )
        
        # 驗證
        self.assertIsInstance(gradients, list)
        self.assertTrue(len(gradients) > 0)
        for grad in gradients:
            self.assertIsInstance(grad, torch.Tensor)
            
    def test_compute_influence_for_pair(self):
        """測試對於排序對的影響力計算"""
        # 創建簡單數據集
        class SimpleRankingDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.x1 = torch.randn(4, 10)
                self.x2 = torch.randn(4, 10)
                self.targets = torch.tensor([1, -1, 1, -1])
                
            def __len__(self):
                return len(self.x1)
                
            def __getitem__(self, idx):
                return (self.x1[idx], self.x2[idx], self.targets[idx], 
                        0, 0, f"sample_a_{idx}", f"sample_b_{idx}")
        
        dataset = SimpleRankingDataset()
        test_x1 = torch.randn(1, 10)
        test_x2 = torch.randn(1, 10)
        test_target = torch.tensor([1])
        
        # 計算影響力
        influence_scores, per_checkpoint_influences = self.tracin.compute_influence_for_pair(
            dataset, test_x1, test_x2, test_target, batch_size=2, num_workers=0
        )
        
        # 驗證
        self.assertIsInstance(influence_scores, dict)
        self.assertEqual(len(influence_scores), 4)  # 4個訓練樣本對
        self.assertIsInstance(per_checkpoint_influences, dict)
        
    def test_compute_self_influence_for_pairs(self):
        """測試排序對的自影響力計算"""
        # 創建簡單數據集
        class SimpleRankingDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.x1 = torch.randn(4, 10)
                self.x2 = torch.randn(4, 10)
                self.targets = torch.tensor([1, -1, 1, -1])
                
            def __len__(self):
                return len(self.x1)
                
            def __getitem__(self, idx):
                return (self.x1[idx], self.x2[idx], self.targets[idx], 
                        0, 0, f"sample_a_{idx}", f"sample_b_{idx}")
        
        dataset = SimpleRankingDataset()
        
        # 計算自影響力
        self_influence_scores, per_checkpoint_self_influences = self.tracin.compute_self_influence_for_pairs(
            dataset, batch_size=2, num_workers=0
        )
        
        # 驗證
        self.assertIsInstance(self_influence_scores, dict)
        self.assertEqual(len(self_influence_scores), 4)  # 4個訓練樣本對
        self.assertIsInstance(per_checkpoint_self_influences, dict)


if __name__ == "__main__":
    unittest.main() 