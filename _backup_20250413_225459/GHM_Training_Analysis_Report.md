# 梯度調和機制 (GHM) 訓練分析報告

## 概述

本報告分析使用梯度調和機制(Gradient Harmonizing Mechanism, GHM)進行模型訓練的過程和結果。GHM是一種優化損失函數的方法，透過平衡梯度分佈來提高模型訓練效果，特別是針對困難樣本的處理能力。

## 訓練參數

- **材質:** plastic
- **頻率:** 3000hz
- **時間戳:** 20250411_232113
- **隨機種子:** 42
- **GHM參數:**
  - Bins: 10
  - Alpha: 0.75
  - Margin: 10.0
- **總訓練輪次:** 60

## 訓練與驗證表現

![訓練與驗證表現](ghm_visualizations/plastic_3000hz_20250411_232113/training_comparison_20250411_232113.png)

*圖1：訓練與驗證表現比較，對比原始損失與GHM損失*

## 梯度分佈分析

GHM通過重新平衡梯度權重，幫助模型更有效地學習。以下是訓練不同階段的梯度分佈可視化：

### 訓練初期(第1輪)

![初期梯度分佈](ghm_visualizations/plastic_3000hz_20250411_232113/epoch1_batch0_20250411_232113.png)

*圖2：訓練初期的梯度分佈*

### 訓練中期(第30輪)

![中期梯度分佈](ghm_visualizations/plastic_3000hz_20250411_232113/epoch30_batch0_20250411_232113.png)

*圖3：訓練中期的梯度分佈*

### 訓練後期(第60輪)

![後期梯度分佈](ghm_visualizations/plastic_3000hz_20250411_232113/epoch60_batch0_20250411_232113.png)

*圖4：訓練後期的梯度分佈*

## GHM統計分析

統計分析梯度分佈的變化可以幫助我們理解GHM的工作原理和效果。以下圖表顯示不同訓練階段的梯度分佈情況：

![GHM統計分析](ghm_visualizations/plastic_3000hz_20250411_232113/stats_analysis/ghm_stats_epoch_1.png)

*圖5：第1輪的GHM統計分析*

![GHM統計分析](ghm_visualizations/plastic_3000hz_20250411_232113/stats_analysis/ghm_stats_epoch_30.png)

*圖6：第30輪的GHM統計分析*

![GHM統計分析](ghm_visualizations/plastic_3000hz_20250411_232113/stats_analysis/ghm_stats_epoch_60.png)

*圖7：第60輪的GHM統計分析*

## 結論與發現

1. **GHM對訓練的影響**：GHM通過重新分配梯度權重，使模型能更好地處理難易程度不同的樣本。
   
2. **梯度分佈演變**：從訓練初期到後期，梯度分佈逐漸變得更均衡，表明GHM能有效減少極端梯度的影響。

3. **準確率提升**：GHM損失函數相比原始損失函數能更有效地提升模型準確率，尤其是在處理難以區分的樣本時。

4. **損失比較**：GHM損失與原始損失的比較顯示，GHM能夠在訓練後期保持更穩定的損失下降趨勢。

5. **最佳模型**：最佳模型檢查點已保存，可用於未來的預測和分析。

## 未來工作

- 嘗試更多GHM參數組合(bins數量、alpha值)以進一步優化效果
- 在不同頻率和材質數據上測試GHM的有效性
- 將GHM與其他優化技術結合，探索更多潛力

## 附錄

完整的分析細節可在`ghm_analysis_report/plastic_3000hz_20250411_232113/analysis_report.md`中找到。
所有可視化資料已保存在`ghm_visualizations/plastic_3000hz_20250411_232113/`目錄。 