# TracIn 模組

這個模組封裝了基於 TracIn 算法的影響力分析功能，用於理解訓練樣本對模型預測的影響。

## 模組結構

```
tracin/
  ├── core/          - 核心實現
  │   ├── tracin.py           - 基礎 TracIn 實現
  │   └── ranking_tracin.py   - 排序特定的 TracIn 實現
  ├── utils/         - 實用工具
  │   ├── influence_utils.py  - 處理影響力分數的工具
  │   └── visualization.py    - 視覺化工具
  ├── scripts/       - 命令行腳本
  │   ├── compute_influence.py   - 計算影響力
  │   ├── generate_exclusions.py - 生成排除列表
  │   ├── analyze.py             - 分析功能
  │   └── visualize.py           - 視覺化功能
  └── tests/         - 測試代碼
```

## 使用方法

### 作為模組使用

```python
from tracin.core.ranking_tracin import RankingTracInCP
from tracin.utils.influence_utils import load_influence_scores

# 初始化 TracIn
tracin = RankingTracInCP(model, checkpoints, loss_type="standard")

# 計算影響力
influence_scores = tracin.compute_influence_for_pair(dataset, test_x1, test_x2, test_target)
```

### 通過命令行使用

```bash
# 計算影響力
python -m tracin.scripts.compute_influence --frequency 500hz --material metal

# 生成排除列表
python -m tracin.scripts.generate_exclusions --metadata-file path/to/metadata.json
```

## 與主訓練流程的整合

TracIn 模組通過以下方式與主訓練流程整合：

1. 生成樣本排除列表
2. 提供影響力分析報告
3. 視覺化訓練樣本的影響力 