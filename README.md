# audio-angle-classification

一個深度學習系統，使用光譜圖和多種神經網絡架構（CNN、ResNet、ConvNeXt）進行音頻角度分類

## 專案介紹

Audio Angle Classification 專案旨在通過分析音頻頻譜圖來精確預測聲音來源的角度。該系統特別適用於需要準確識別聲音方向的場景，如聲學定位、智能音響系統和機器人聽覺感知。

### 核心技術

- **音頻頻譜分析**：將原始音頻轉換為頻譜圖以進行深度學習處理
- **排序學習**：使用成對比較方法來學習角度的排序關係
- **Gradient Harmonizing Mechanism (GHM)**：特殊的損失函數，能更有效地處理困難樣本
- **視覺化分析工具**：提供訓練過程中各種指標的視覺化，尤其是 GHM 統計數據的動態變化

### 主要特點

- 支持多種網絡架構（輕量級 CNN、ResNet 變體）
- 支持不同材質和頻率的音頻數據處理
- 實現了創新的 GHM 損失函數，提高對樣本不平衡的抵抗力
- 提供豐富的視覺化和分析工具

## 安裝指南

### 依賴項

本專案需要以下主要依賴：

```
torch>=1.8.0
numpy>=1.19.0
matplotlib>=3.3.0
```

### 安裝步驟

1. 克隆此倉庫：

```bash
git clone https://github.com/sk413025/audio-angle-classification.git
cd audio-angle-classification
```

2. 創建並啟用虛擬環境（推薦）：

```bash
conda create -n audio-class python=3.8
conda activate audio-class
```

3. 安裝依賴：

```bash
pip install -r requirements.txt
```

## 使用說明

### 數據準備

將音頻數據放置於 `data` 目錄，結構如下：

```
data/
  ├── {class_name}/
  │     ├── {material}/
  │     │     ├── {material}_{class_name}_{frequency}_{seq_num}.wav
```

例如：`data/0_degree/plastic/plastic_0_degree_1000hz_1.wav`

### 模型訓練

#### 標準訓練

使用標準 MarginRankingLoss 進行訓練：

```bash
python train.py --frequency 1000hz --material plastic --loss-type standard
```

#### GHM 訓練

使用 Gradient Harmonizing Mechanism 損失進行訓練：

```bash
python train.py --frequency 1000hz --material plastic --loss-type ghm --ghm-bins 10 --ghm-alpha 0.75
```

參數說明：
- `--frequency`：選擇使用的頻率數據（500hz, 1000hz, 3000hz, all）
- `--material`：選擇材質類型
- `--loss-type`：選擇損失函數類型（standard, ghm）
- `--ghm-bins`：GHM 分箱數量
- `--ghm-alpha`：GHM alpha 參數

### 模型評估

評估模型性能：

```bash
python evaluate.py --model-path saved_models/model_checkpoints/plastic_1000hz_ghm_20250414_012345/model_epoch_30.pt
```

### 視覺化工具使用

#### GHM 視覺化分析

可視化單個訓練統計：

```python
from utils.visualization import visualize_ghm_stats

visualize_ghm_stats(
    "saved_models/stats/plastic_1000hz_ghm_20250414_011143/epoch30_batch0_20250414_011143.npy",
    output_path="visualization.png"
)
```

比較不同 epoch 的 GHM 統計：

```python
from utils.visualization import compare_ghm_epochs

compare_ghm_epochs(
    "saved_models/stats/plastic_1000hz_ghm_20250414_011143",
    epochs=[1, 10, 20, 30],
    output_path="epoch_comparison.png"
)
```

分析整個訓練過程：

```python
from utils.visualization import analyze_ghm_training

analyze_ghm_training(
    "saved_models/stats/plastic_1000hz_ghm_20250414_011143",
    output_dir="analysis_output"
)
```

## 專案架構

```
audio-angle-classification/
  ├── train.py                 # 主訓練腳本
  ├── datasets.py              # 數據集定義
  ├── config.py                # 配置參數
  ├── models/                  # 模型定義
  │     ├── resnet_ranker.py   # ResNet 和 CNN 模型
  ├── losses/                  # 損失函數
  │     ├── ghm_loss.py        # GHM 損失函數實現
  ├── utils/                   # 工具和輔助函數
  │     ├── common_utils.py    # 通用工具函數
  │     ├── debugging_utils.py # 調試工具
  │     ├── ghm_utils.py       # GHM 相關工具
  │     ├── visualization/     # 視覺化工具
  │           ├── ghm_analyzer.py     # GHM 視覺化分析工具
  │           ├── plot_utils.py       # 圖表繪製工具
  ├── saved_models/            # 保存的模型和統計數據
```

### 核心模組

- **模型模組**：定義了音頻角度分類的神經網絡架構
- **損失函數模組**：實現標準和 GHM 損失函數
- **數據集模組**：處理音頻頻譜數據的加載和預處理
- **視覺化模組**：提供訓練與評估過程的視覺化工具

### 視覺化模組功能

視覺化模組提供了多種功能來幫助理解模型訓練過程：

- 訓練歷史視覺化
- GHM 分箱統計分析
- 跨 epoch 的 GHM 變化比較
- 梯度分佈可視化

## 模型與數據集

### 支援的模型架構

- **SimpleCNNAudioRanker**：輕量級 CNN 架構，適合資源受限場景
- **ResNetAudioRanker** (計劃中)：基於 ResNet 的架構，用於更複雜的場景

### 數據集格式

專案使用音頻頻譜圖數據集，支持以下格式：

- 音頻格式：WAV 文件
- 頻率選項：500Hz, 1000Hz, 3000Hz
- 材質選項：plastic, metal, wood 等
- 角度類別：多個角度類別（如 0_degree, 45_degree 等）

### 預處理流程

1. 讀取 WAV 音頻文件
2. 轉換為單聲道（如果是立體聲）
3. 計算短時傅立葉變換 (STFT)
4. 轉換為分貝刻度的頻譜圖
5. 標準化處理

## 實驗結果

### 主要發現

- GHM 損失函數相較標準損失函數能有效提高困難樣本的分類準確率
- 在不平衡數據集上，GHM 訓練的模型表現更加穩定
- 1000Hz 頻率數據在多數場景下表現最佳

### 性能比較

| 模型           | 損失函數      | 準確率 | 備註                     |
|---------------|--------------|-------|-------------------------|
| SimpleCNN     | Standard     | 85%   | 基準模型                  |
| SimpleCNN     | GHM          | 91%   | 在困難樣本上表現更好        |

### 實驗報告

詳細實驗結果請參考我們的報告：

- [GHM 訓練分析報告](experiments/20250411_GHM_Training_Analysis/README.md) - Gradient Harmonizing Mechanism 訓練效果分析 (2025-04-11)

## 參與貢獻

我們歡迎各種形式的貢獻，包括但不限於：

- 報告 Bug
- 提交功能需求
- 提供代碼改進
- 完善文檔

### 貢獻步驟

1. Fork 此倉庫
2. 創建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 創建一個 Pull Request

## 許可證

本專案採用 MIT 許可證 - 詳見 [LICENSE](LICENSE) 文件
