# TracIn: 訓練數據影響力評估工具

本工具基於論文《Estimating Training Data Influence by Tracing Gradient Descent》實現，用於評估訓練數據對模型預測結果的影響程度。

## 功能概述

- **自影響力計算**: 識別訓練集中的困難樣本和異常值
- **影響力計算**: 評估訓練樣本對測試樣本預測的影響
- **多設備支持**: 兼容CUDA、MPS (MacOS) 和CPU設備
- **多種損失函數**: 支持標準MarginRankingLoss和GHMRankingLoss

## 安裝與依賴

本工具依賴以下Python庫:
- PyTorch (>=1.7.0)
- NumPy
- 其他專案內部模組

## 使用方法

### 計算自影響力

自影響力反映了訓練樣本的「難度」，計算方法如下:

```bash
python compute_tracin_influence.py \
    --frequency 500hz \
    --material plastic \
    --checkpoint-dir saved_models/model_checkpoints/your_checkpoint_dir/ \
    --compute-self-influence \
    --loss-type standard \
    --device mps
```

### 計算影響力

影響力反映了訓練樣本對測試樣本預測的影響程度，計算方法如下:

```bash
python compute_tracin_influence.py \
    --frequency 500hz \
    --material plastic \
    --checkpoint-dir saved_models/model_checkpoints/your_checkpoint_dir/ \
    --compute-influence \
    --num-test-samples 5 \
    --device mps
```

### 參數說明

| 參數 | 描述 | 默認值 |
|------|------|--------|
| `--frequency` | 頻率數據 (500hz, 1000hz, 3000hz, all) | 必填 |
| `--material` | 材料類型 | 從config.py中獲取 |
| `--checkpoint-dir` | 模型檢查點目錄 | 必填 |
| `--checkpoint-prefix` | 檢查點文件前綴 | model_epoch_ |
| `--metadata-dir` | 保存元數據的目錄 | /Users/sbplab/Hank/sinetone_sliced/.../metadata |
| `--loss-type` | 損失函數類型 (standard, ghm) | standard |
| `--margin` | 排序損失的邊界參數 | 1.0 |
| `--ghm-bins` | GHM損失的bins數量 | 10 |
| `--ghm-alpha` | GHM損失的alpha參數 | 0.75 |
| `--batch-size` | 計算梯度的批次大小 | 16 |
| `--num-workers` | 數據加載的工作線程數 | 4 |
| `--seed` | 隨機種子 | 42 |
| `--compute-self-influence` | 計算自影響力 | False |
| `--compute-influence` | 計算影響力 | False |
| `--num-test-samples` | 計算影響力的測試樣本數量 | 5 |
| `--device` | 計算設備 (cpu, cuda, mps) | 自動檢測 |

## 結果解讀

計算結果會以JSON格式保存在指定的元數據目錄中:

### 自影響力文件

```json
{
  "sample_id1_sample_id2": {
    "tracin_self_influence": score_value
  },
  ...
}
```

### 影響力文件

```json
{
  "train_sample_id1_train_sample_id2": {
    "tracin_influence_test_sample_id1_test_sample_id2": score_value
  },
  ...
}
```

## 應用場景

1. **數據質量審核**: 識別並審核高自影響力的樣本，可能存在標註錯誤
2. **數據增強策略**: 基於不同角度類別的影響力分布，定向增強數據
3. **模型優化**: 調整損失函數或模型結構，更好地處理高影響力樣本
4. **預測解釋**: 了解模型預測依賴哪些訓練樣本，增強可解釋性

## 高級使用

詳細的分析報告和解讀指南請參見 `tracin_analysis_report.md`。

## 參考資料

- [原始論文](https://arxiv.org/abs/2002.08484)
- 代碼實現參考了[TracIn官方實現](https://github.com/frederick0329/TracIn) 