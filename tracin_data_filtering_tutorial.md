# 使用TracIn分析改進數據集質量

本教程介紹如何結合使用TracIn方法和數據集管理功能來識別和排除對模型泛化能力有負面影響的訓練樣本。

## 概述

整個過程包括以下步驟：

1. 使用TracIn計算影響力分數
2. 分析影響力分數，識別有害樣本
3. 生成排除列表
4. 使用排除列表重新訓練模型

## 步驟1：計算TracIn影響力分數

首先，我們需要計算訓練樣本對測試樣本的影響力分數。這裡我們使用TracIn方法，它通過分析訓練過程中的參數梯度來估計每個訓練樣本對模型預測的影響。

使用以下命令計算影響力分數：

```bash
python compute_tracin_influence.py \
    --frequency 500hz \
    --material plastic \
    --checkpoint-dir /Users/sbplab/Hank/sinetone_sliced/step_018_sliced/checkpoints/plastic_500hz_standard \
    --checkpoint-prefix model_checkpoint_epoch \
    --metadata-dir /Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata \
    --compute-influence
```

這將生成包含影響力分數的元數據文件，保存在指定的元數據目錄中：`/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_influence_metadata.json`。

## 步驟2：分析影響力分數並生成排除列表

接下來，我們使用`generate_tracin_exclusions.py`腳本來分析影響力分數，識別對泛化能力有負面影響的訓練樣本，並生成排除列表：

```bash
python generate_tracin_exclusions.py \
    --metadata-file /Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_influence_metadata.json \
    --output-file /Users/sbplab/Hank/sinetone_sliced/step_018_sliced/exclusions/plastic_500hz_tracin_exclusions.txt \
    --threshold -3.0 \
    --min-occurrences 5 \
    --max-exclusions 30 \
    --consider-both-samples \
    --verbose \
    --update-metadata
```

參數說明：
- `--metadata-file`：TracIn影響力元數據文件路徑
- `--output-file`：生成的排除列表輸出路徑
- `--threshold`：負面影響力閾值（低於此值的樣本將被排除）
- `--min-occurrences`：一個樣本至少在多少個測試樣本上有負面影響才被排除
- `--max-exclusions`：最大排除樣本數量
- `--consider-both-samples`：同時考慮訓練對中的兩個樣本
- `--verbose`：顯示詳細輸出
- `--update-metadata`：將排除原因更新到樣本元數據文件中

### 追蹤排除原因

當使用`--update-metadata`參數時，腳本會將排除的樣本及其排除原因記錄到相應的樣本元數據文件中（例如`plastic_500hz_metadata.json`）。這對於以下幾點非常有用：

1. **可追溯性**：記錄為什麼某個樣本被排除，方便後續審查
2. **數據分析**：可以分析被排除樣本的共同特徵
3. **模型診斷**：了解模型對哪類樣本敏感

元數據文件中會記錄以下信息：
- 樣本是否被排除（`excluded`字段）
- 排除原因和時間（`notes`字段）
- 詳細的TracIn分析結果（`tracin_info`字段）
  - 排除日期
  - 負面影響出現次數
  - 平均影響力分數
  - 受影響的測試樣本列表

您可以使用以下命令查看更新的元數據：

```bash
python -c "import json; f=open('/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_metadata.json'); m=json.load(f); print(json.dumps([v for k,v in m.items() if v.get('excluded')], indent=2))"
```

## 步驟3：使用排除列表重新訓練模型

最後，我們使用生成的排除列表來重新訓練模型。在`train.py`中加入`--exclusions-file`參數：

```bash
python train.py \
    --frequency 500hz \
    --material plastic \
    --exclusions-file /Users/sbplab/Hank/sinetone_sliced/step_018_sliced/exclusions/plastic_500hz_tracin_exclusions.txt
```

### 排除過程的工作原理

排除列表是一個簡單的文本文件，每行包含一個要排除的樣本ID。訓練過程中，`datasets`模組的`AudioSpectrumDataset`類會讀取此文件，並在數據加載過程中過濾掉這些樣本：

1. `DatasetConfig`類加載排除列表
2. `ManagedDataset`基類維護有效樣本的索引列表
3. `AudioSpectrumDataset`使用這些索引來過濾樣本對，確保任何包含被排除樣本的對都不會被用於訓練

## 高級用法

### 分析元數據中的排除模式

您可以使用以下腳本分析元數據，發現排除樣本的模式：

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# 加載元數據
with open('/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata/plastic_500hz_metadata.json', 'r') as f:
    metadata = json.load(f)

# 提取被排除的樣本
excluded_samples = {k: v for k, v in metadata.items() if v.get('excluded')}

# 分析角度分佈
angles = [sample['angle'] for sample in excluded_samples.values()]
plt.hist(angles, bins=36)
plt.title('排除樣本的角度分佈')
plt.xlabel('角度')
plt.ylabel('樣本數')
plt.savefig('excluded_angles.png')

# 分析影響力分數
influence_scores = [sample.get('tracin_info', {}).get('average_influence', 0) 
                    for sample in excluded_samples.values()]
plt.figure()
plt.hist(influence_scores, bins=20)
plt.title('排除樣本的平均影響力分數')
plt.xlabel('影響力分數')
plt.ylabel('樣本數')
plt.savefig('excluded_influence_scores.png')
```

### 結合TracIn排除與GHM損失

TracIn排除與GHM損失函數配合使用效果更佳。GHM關注每個批次中的困難樣本，而TracIn識別長期對泛化有害的樣本：

```bash
python train.py \
    --frequency 500hz \
    --material plastic \
    --exclusions-file /Users/sbplab/Hank/sinetone_sliced/step_018_sliced/exclusions/plastic_500hz_tracin_exclusions.txt \
    --loss-type ghm
```

## 結論

通過結合TracIn影響力分析與現有的數據集管理功能，我們可以實現以下目標：

1. 識別並排除對模型泛化能力有害的訓練樣本
2. 提高模型在測試集上的性能
3. 更好地理解模型的訓練動態
4. 優化數據集質量

這種方法特別適合處理有噪聲或存在問題的數據，可以顯著提高模型的穩健性和泛化能力。 