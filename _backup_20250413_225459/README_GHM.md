# Gradient Harmonizing Mechanism (GHM) for Ranking Tasks

## 概述

這個實現包含了針對排序任務的梯度調和機制 (Gradient Harmonizing Mechanism, GHM)。GHM 是一種用於處理樣本不平衡問題的技術，最初為目標檢測任務提出，在這裡我們將其應用於排序任務。

## 背景與動機

在深度學習任務中，訓練樣本通常存在不平衡問題。例如，在排序任務中，可能存在：
- 大量容易區分的樣本對（差異很大）
- 少量難以區分的樣本對（差異很小）

傳統的損失函數（如 MarginRankingLoss）對所有樣本一視同仁，導致容易樣本主導梯度，使模型無法充分學習困難樣本。GHM 通過分析梯度分佈自動調整樣本權重，使不同難度的樣本都能對模型訓練做出適當貢獻。

## 實現內容

本實現包含以下檔案：

1. `ghm_loss.py` - 包含 GHM 損失函數實現
   - `GHMCLoss` - 用於分類任務的 GHM
   - `GHMRLoss` - 用於回歸任務的 GHM
   - `GHMRankingLoss` - 針對排序任務的 GHM，可替代 MarginRankingLoss

2. `ghm_utils.py` - GHM 相關的工具函數
   - 繪製梯度分佈和權重
   - 計算 GHM 統計數據
   - 比較 GHM 和常規損失

3. `train_with_ghm.py` - 使用 GHM 訓練模型的示例腳本
   - 與原始訓練腳本功能相同
   - 使用 GHM 損失函數
   - 記錄和比較 GHM 與傳統損失函數的效果

4. `test_ghm.py` - 測試 GHM 實現的腳本
   - 基本功能測試
   - 不平衡數據處理測試
   - 訓練效果比較測試

## 使用方法

### 1. 測試 GHM 實現

運行測試腳本，檢查 GHM 的基本功能和效果：

```bash
python test_ghm.py
```

測試結果會保存在 `./test_results` 目錄下。

### 2. 使用 GHM 訓練模型

運行 GHM 訓練腳本，按照提示選擇頻率和設置參數：

```bash
python train_with_ghm.py
```

### 3. 在自己的代碼中使用 GHM

將 `ghm_loss.py` 添加到你的項目中，然後像這樣使用：

```python
from ghm_loss import GHMRankingLoss

# 創建 GHM 損失
ghm_criterion = GHMRankingLoss(
    margin=0.3,    # 與 MarginRankingLoss 相同的 margin 參數
    bins=10,       # 梯度密度直方圖的 bin 數量
    alpha=0.75     # 梯度密度調節指數
)

# 在訓練循環中使用
outputs1 = model(data1)
outputs2 = model(data2)
loss = ghm_criterion(outputs1, outputs2, targets)
loss.backward()
optimizer.step()
```

## 參數說明

`GHMRankingLoss` 有以下參數：

- `margin`: 與 MarginRankingLoss 相同的邊界參數
- `bins`: 梯度密度直方圖的 bin 數量，一般設置為 10-30
- `alpha`: 梯度密度調節指數，控制重新加權的強度，一般為 0.5-1.0

## 優勢

使用 GHM 可以獲得以下優勢：

1. **自動樣本重新加權** - 不需要手動設置範例權重
2. **專注於有信息量的樣本** - 不會被大量簡單樣本或極少數極難樣本主導
3. **穩定的梯度分佈** - 不同難度的樣本得到平衡的梯度貢獻
4. **無需精確調參** - 相比 Focal Loss 等方法，參數不那麼敏感

## 參考文獻

- Li et al. "Gradient Harmonized Single-stage Detector." AAAI 2019.
- 原始論文：https://arxiv.org/abs/1811.05181 