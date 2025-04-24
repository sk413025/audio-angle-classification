# TracIn 模組 (已整合到 datasets 模組)

這個模組封裝了基於 TracIn 算法的影響力分析功能，用於理解訓練樣本對模型預測的影響。

## 模組結構

```
datasets/tracin/
  ├── core/          - 核心實現
  │   ├── tracin.py           - 基礎 TracIn 實現
  │   └── ranking_tracin.py   - 排序特定的 TracIn 實現
  ├── utils/         - 實用工具
  │   ├── influence_utils.py  - 處理影響力分數的工具
  │   └── visualization.py    - 視覺化工具
  ├── scripts/       - 命令行腳本
  │   ├── compute_influence.py   - 計算影響力
  │   ├── generate_exclusions.py - 生成排除列表
  │   ├── test_pair_exclusion.py - 測試排除閾值
  │   ├── exclude_harmful_pairs.sh - 排除有害 pair 工作流程
  └── tests/         - 測試代碼
```

## 使用方法

### 作為模組使用

```python
from datasets.tracin.core.ranking_tracin import RankingTracInCP
from datasets.tracin.utils.influence_utils import load_influence_scores

# 初始化 TracIn
tracin = RankingTracInCP(model, checkpoints, loss_type="standard")

# 計算影響力
influence_scores = tracin.compute_influence_for_pair(dataset, test_x1, test_x2, test_target)
```

### 通過命令行使用

```bash
# 計算影響力
python -m datasets.tracin.scripts.compute_influence --frequency 500hz --material metal

# 生成排除列表
python -m datasets.tracin.scripts.generate_exclusions --metadata-file path/to/metadata.json
```

## 排除有害 Ranking Pair

TracIn 可以識別對模型泛化能力有負面影響的 ranking pair，並將其排除於訓練之外。這個功能利用了現有的樣本排除機制，不需要修改 `datasets` 模組。

### 排除有害 Pair 的工作流程

1. **計算 TracIn 影響力分數**：使用 `compute_influence.py` 計算訓練樣本的影響力
2. **分析和測試閾值**：使用 `test_pair_exclusion.py` 評估不同閾值下排除的 pair 數量
3. **生成排除列表**：使用 `generate_exclusions.py` 生成排除列表
4. **使用排除列表進行訓練**：使用標準的 `train.py` 帶上 `--exclusions-file` 參數

### 排除模式

* **ranking_pair** (默認)：排除特定的 ranking pair 組合，但保留單個樣本
* **full_pair**：排除構成有害 pair 的兩個樣本

### 使用範例

#### 步驟 1: 計算 TracIn 影響力

```bash
python -m datasets.tracin.scripts.compute_influence \
  --frequency 500hz \
  --material plastic \
  --checkpoint-dir /path/to/checkpoints \
  --compute-influence \
  --num-test-samples 5
```

#### 步驟 2: 測試閾值影響

```bash
python -m datasets.tracin.scripts.test_pair_exclusion \
  --metadata-file /path/to/metadata/plastic_500hz_influence_metadata.json \
  --thresholds -5.0,-10.0,-15.0
```

#### 步驟 3: 生成排除列表

```bash
python -m datasets.tracin.scripts.generate_exclusions \
  --metadata-file /path/to/metadata/plastic_500hz_influence_metadata.json \
  --output-file /path/to/exclusions/plastic_500hz_excluded_pairs.txt \
  --threshold -5.0 \
  --pair-mode ranking_pair
```

#### 步驟 4: 使用排除列表進行訓練

```bash
python train.py \
  --frequency 500hz \
  --material plastic \
  --exclusions-file /path/to/exclusions/plastic_500hz_excluded_pairs.txt
```

### 便捷工作流程腳本

使用 `exclude_harmful_pairs.sh` 一次性完成整個工作流程：

```bash
./datasets/tracin/scripts/exclude_harmful_pairs.sh \
  --frequencies 500hz,1000hz \
  --threshold -5.0 \
  --pair-mode ranking_pair
```

查看所有選項：

```bash
./datasets/tracin/scripts/exclude_harmful_pairs.sh --help
```

## 與主訓練流程的整合

TracIn 模組通過以下方式與主訓練流程整合：

1. 生成樣本排除列表
2. 提供影響力分析報告
3. 視覺化訓練樣本的影響力 