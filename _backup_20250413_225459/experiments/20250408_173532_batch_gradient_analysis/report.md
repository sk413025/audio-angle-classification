# 實驗報告: 20250408_173532_batch_gradient_analysis

## 1. 元數據 (Metadata)

* **實驗編號 (Experiment ID):** 20250408_173532_batch_gradient_analysis
* **日期時間 (Timestamp):** 2025-04-08 17:35:32
* **實驗描述 (Description):** 批次梯度分析與視覺化，探討不同訓練階段批次梯度的行為特性
* **執行者 (Executor):** Claude 3.7 Sonnet

## 2. 背景動機 (Background & Motivation)

在深度學習模型訓練過程中，梯度的行為對優化過程有著至關重要的影響。批次梯度噪聲尺度（Gradient Noise Scale, GNS）作為一個關鍵指標，衡量不同批次之間梯度方向的差異程度。這種差異可能揭示數據集潛在問題或表明模型訓練中的特定階段行為。

研究表明，批次梯度的一致性與模型收斂和泛化能力息息相關。本實驗旨在通過分析不同訓練階段的批次梯度特性，深入理解我們的音頻分類模型在訓練過程中的梯度行為，從而優化訓練策略和評估數據質量。

## 3. 實驗目的與假設 (Purpose & Hypothesis)

**實驗目的：**
1. 監測 ResNetAudioRanker 模型在訓練過程中不同階段（epoch 5, 10, 15, 20, 25, 30）的梯度行為
2. 分析批次梯度噪聲尺度(GNS)隨訓練進度的變化趨勢
3. 通過梯度方向的 t-SNE 視覺化，評估不同批次梯度的分布特性
4. 通過批次梯度余弦相似度矩陣，分析批次間梯度方向的一致性

**假設：**
1. 隨著訓練進行，GNS 值應該逐漸降低，表明模型漸漸收斂
2. 訓練後期的梯度方向應該比早期更加一致，呈現更緊湊的聚類
3. 若存在批次梯度方向的顯著差異或異常聚類，可能暗示數據集中存在問題樣本

## 4. 方法 (Methodology)

### 使用的數據集
- **SpectrogramDatasetWithMaterial**：角度分類音頻光譜圖數據集
- **材質類型**：plastic
- **頻率選擇**：3000Hz
- **樣本總數**：54個樣本，6個不同角度類別（deg000, deg036, deg072, deg108, deg144, deg180）

### 數據預處理與模型架構
1. **數據預處理**：將原始音頻數據轉換為 STFT 光譜圖
2. **數據分割**：90% 訓練集，10% 驗證集
3. **模型架構**：ResNetAudioRanker（基於 ResNet-18 主幹）
4. **訓練方式**：排序對（ranking pair）訓練，使用 MarginRankingLoss

### 梯度分析方法
1. **批次梯度計算**：對每個批次數據單獨計算梯度向量
2. **梯度噪聲尺度(GNS)計算**：評估批次梯度與平均梯度的偏差程度
3. **t-SNE 視覺化**：將高維梯度向量降維至二維空間進行視覺化
4. **余弦相似度矩陣**：計算批次間梯度方向的相似性

### 實驗環境
- **硬體**：Mac Studio (MPS 加速)
- **軟體**：PyTorch 框架
- **超參數**：
  - 批次大小: 4
  - 學習率: 0.001
  - 權重衰減: 1e-5
  - 排名邊界值: 10.0

## 5. 結果與分析 (Results & Analysis)

### 5.1 梯度噪聲尺度(GNS)隨訓練進度的變化

在分析 3000Hz 頻率下的模型訓練過程中，我們觀察到 GNS 值的變化趨勢：

| Epoch | GNS 值    |
|-------|-----------|
| 5     | 0.000017  |
| 10    | 0.000015  |
| 15    | 0.000006  |
| 20    | 0.000009  |
| 25    | 0.000009  |
| 30    | 0.000011  |

![GNS vs Epoch](./gns_vs_epoch_plastic_3000hz.png)

從 GNS 值的變化趨勢可以看出：
- GNS 整體呈現先下降後輕微上升的趨勢
- 在 Epoch 15 達到最低點 (0.000006)
- 從 Epoch 15 到 Epoch 30，GNS 值略有上升但保持在較低水平

**分析**：GNS 值整體非常低（數量級在 10^-5 至 10^-6 之間），表明不同批次的梯度方向極為一致。這種一致性暗示模型訓練過程穩定，數據集質量良好。GNS 在 Epoch 15 達到最低點後略有上升，可能表明模型開始專注於更難的樣本或細微的優化方向。

### 5.2 梯度方向的 t-SNE 視覺化分析

通過 t-SNE 降維，我們可以直觀地觀察不同訓練階段批次梯度方向的分布：

**早期階段 (Epoch 5)：**
梯度方向呈現相對分散的聚類，部分批次梯度方向存在較大差異。

![Epoch 5 t-SNE](./images/gradient_directions_tsne_epoch_5.png)

**中期階段 (Epoch 15)：**
梯度方向形成更緊湊的聚類，大多數批次梯度方向更加一致。

![Epoch 15 t-SNE](./images/gradient_directions_tsne_epoch_15.png)

**後期階段 (Epoch 30)：**
梯度方向聚類非常緊湊，表明模型已經收斂到穩定狀態。

![Epoch 30 t-SNE](./images/gradient_directions_tsne_epoch_30.png)

**分析**：從 t-SNE 視覺化結果可以看出，隨著訓練的進行，批次梯度方向的一致性明顯增強，聚類更加緊湊，這與 GNS 值的變化趨勢基本一致。此外，沒有觀察到明顯的離群點或異常聚類，表明數據集中沒有顯著的問題樣本。

### 5.3 批次梯度余弦相似度矩陣分析

余弦相似度矩陣顯示了不同批次梯度方向之間的相似性：

**早期階段 (Epoch 5)：**
相似度矩陣中存在一些較低相似度的區域，表明部分批次梯度方向差異較大。

![Epoch 5 相似度矩陣](./images/gradient_similarities_epoch_5.png)

**中期階段 (Epoch 15)：**
相似度矩陣呈現高度一致的淺色區域，表明不同批次梯度方向高度相似。

![Epoch 15 相似度矩陣](./images/gradient_similarities_epoch_15.png)

**後期階段 (Epoch 30)：**
相似度矩陣顯示幾乎所有批次梯度方向都高度一致。

![Epoch 30 相似度矩陣](./images/gradient_similarities_epoch_30.png)

**分析**：余弦相似度矩陣結果進一步確認了 GNS 和 t-SNE 分析的結論。隨著訓練進行，批次間梯度方向的一致性顯著提高，模型逐漸收斂到一個穩定狀態。矩陣中沒有出現明顯的負相關區域，這表明不同批次的訓練目標一致，數據中不存在明顯衝突。

## 6. 結論與未來工作 (Conclusion & Future Work)

### 6.1 結論

1. **數據質量評估**：
   - 極低的 GNS 值（10^-5 至 10^-6 量級）表明數據集質量良好，不存在顯著的衝突樣本。
   - 緊湊的梯度聚類和高度一致的相似度矩陣進一步支持這一結論。

2. **模型訓練穩定性**：
   - GNS 值在訓練過程中呈現先下降後小幅上升的趨勢，總體保持在非常低的水平。
   - 梯度方向的分布從相對分散逐漸變得緊湊，表明優化過程穩定。

3. **批次大小評估**：
   - 當前批次大小 (4) 對於這個數據集和模型架構是合適的，GNS 值極低表明不需要更大的批次大小。
   - 考慮到 GNS 值在後期略有上升，稍微減小批次大小可能有助於引入適當的隨機性，提高泛化能力。

### 6.2 未來工作

1. **不同頻率的比較分析**：
   - 對比 500Hz、1000Hz 和 3000Hz 三種頻率下模型的梯度行為差異。
   - 分析不同頻率數據集的 GNS 值和梯度聚類特性，評估哪種頻率提供的信號更加一致和有用。

2. **批次大小敏感性研究**：
   - 嘗試不同的批次大小（2, 4, 8, 16），分析其對 GNS 值和模型性能的影響。
   - 探索在不同訓練階段動態調整批次大小的策略。

3. **學習率策略優化**：
   - 結合 GNS 分析結果，設計更適合的學習率調度方案。
   - 探索在 GNS 值開始上升時增加學習率的策略，以幫助逃離潛在的局部最小值。

4. **異常樣本檢測**：
   - 開發基於梯度行為的異常樣本檢測方法，識別可能對訓練產生負面影響的樣本。
   - 結合梯度分析和典型特徵分析，建立更全面的數據質量評估框架。

5. **多材質比較研究**：
   - 對比不同材質（plastic, box）下模型的梯度行為差異。
   - 分析材質特性對模型訓練穩定性的影響。 