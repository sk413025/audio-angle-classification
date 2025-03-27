"""
訓練與評估模組：包含訓練和評估相關函數
功能：
- 實現排名模型的訓練邏輯
- 通過二元比較模式實現角度排序
- 計算排序準確率
- 評估模型性能
- 提供訓練過程信息輸出
"""

import torch
from torch.utils.data import DataLoader
from config import BATCH_SIZE, DEVICE

def train_ranker(model, dataloader, optimizer, criterion, device, epoch):
    """
    訓練排名模型
    使用二元比較 (x1, x2, target) 格式的數據
    """
    model.train()
    running_loss = 0.0
    ranking_correct = 0
    total_pairs = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        # 防止批次索引錯誤 - 明確分離所有元素
        data1, data2, target, label1, label2 = batch_data
        # if batch_idx == 0:
        #     import ipdb; ipdb.set_trace()
        
        # 檢查數據是否已經在正確設備上
        if batch_idx == 0 and epoch == 0:
            print(f"Data device check: {data1.device} -> {device}")
            print(f"Data shapes: data1={data1.shape}, data2={data2.shape}, target={target.shape}")
        
        # 修正: 確保目標是正確的形狀和類型
        if target.dim() > 1:
            print(f"WARNING: Target has wrong shape {target.shape}, reshaping...")
            # 如果目標維度錯誤，重建目標張量
            # 根據標籤比較重新建立目標值
            target = (label1 > label2).float() * 2 - 1  # 1 if label1 > label2, else -1
        
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 獲取模型輸出的排序分數
        score1 = model(data1)  # [batch_size, 1]
        score2 = model(data2)  # [batch_size, 1]
        
        # 調整維度
        score1 = score1.squeeze()  # [batch_size]
        score2 = score2.squeeze()  # [batch_size]

        # if batch_idx <= 5:
        #     import ipdb; ipdb.set_trace()
        
        # 計算損失:
        # MarginRankingLoss(x1, x2, target):
        # 如果 target=1，則期望 x1 > x2
        # 如果 target=-1，則期望 x2 > x1
        loss = criterion(score1, score2, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 計算排序準確率:
        # 當 target=1 時，應該是 score1 > score2
        # 當 target=-1 時，應該是 score2 > score1
        correct_predictions = ((target == 1) & (score1 > score2)) | ((target == -1) & (score1 < score2))
        ranking_correct += torch.sum(correct_predictions).item()
        total_pairs += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data1)}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_ranking_acc = ranking_correct / total_pairs if total_pairs > 0 else 0
    
    return epoch_loss, epoch_ranking_acc

def evaluate_ranker(model, dataloader, criterion, device):
    """
    評估排名模型
    使用二元比較 (x1, x2, target) 格式的數據
    """
    model.eval()
    running_loss = 0.0
    ranking_correct = 0
    total_pairs = 0
    
    # with torch.no_grad():
    for batch_idx, batch_data in enumerate(dataloader):
        # 防止批次索引錯誤 - 明確分離所有元素
        data1, data2, target, label1, label2 = batch_data
        # if batch_idx == 0:
        #     import ipdb; ipdb.set_trace()
        
        # 修正: 確保目標是正確的形狀和類型
        if target.dim() > 1:
            # 如果目標維度錯誤，重建目標張量
            target = (label1 > label2).float() * 2 - 1  # 1 if label1 > label2, else -1
            
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        
        # 獲取排序分數
        score1 = model(data1)  # [batch_size, 1]
        score2 = model(data2)  # [batch_size, 1]

        # if batch_idx <= 5:
        #     import ipdb; ipdb.set_trace()
        
        # 調整維度
        score1 = score1.squeeze()  # [batch_size]
        score2 = score2.squeeze()  # [batch_size]
        
        # 計算損失
        loss = criterion(score1, score2, target)
        
        running_loss += loss.item()
        
        # 計算排序準確率
        correct_predictions = ((target == 1) & (score1 > score2)) | ((target == -1) & (score1 < score2))
        ranking_correct += torch.sum(correct_predictions).item()
        total_pairs += target.size(0)


        loss = criterion(score1, score2, target)
    
        
        running_loss += loss.item()


    epoch_loss = running_loss / len(dataloader)
    epoch_ranking_acc = ranking_correct / total_pairs if total_pairs > 0 else 0
    
    return epoch_loss, epoch_ranking_acc

def _create_ranking_pairs(self):
    pairs = []
    # 獲取所有樣本及標籤
    labels = []
    indices = []
    for idx in range(len(self.dataset)):
        _, label = self.dataset[idx]
        label = label.item() if isinstance(label, torch.Tensor) else label
        labels.append(label)
        indices.append(idx)
    
    # 建立索引和標籤的列表
    all_indices = np.array(indices)
    all_labels = np.array(labels)
    
    # 決定要生成多少對
    num_pairs = len(indices) * 10  # 可以調整這個數字來控制對數
    
    # 完全隨機生成配對
    for _ in range(num_pairs):
        # 隨機選擇兩個不同的索引
        i, j = np.random.choice(len(indices), 2, replace=False)
        idx1, idx2 = all_indices[i], all_indices[j]
        label1, label2 = all_labels[i], all_labels[j]
        
        # 根據類別順序確定目標值
        if label1 > label2:
            pairs.append((idx1, idx2, 1))
        elif label1 < label2:
            pairs.append((idx1, idx2, -1))
        # 如果標籤相同，則不添加此對
    
    return pairs
