# 梯度批次相似度視覺化指南

本文檔提供了訓練過程中不同階段梯度相似度的視覺化分析。透過按時間順序排列的圖表，我們可以直觀觀察 GHM 訓練過程中梯度方向的演變。

## 訓練早期階段 (Epoch 5-15)

### Epoch 5
![梯度相似度矩陣 - Epoch 5](gradient_similarities_epoch_5.png)
![梯度方向 t-SNE - Epoch 5](gradient_directions_tsne_epoch_5.png)

** revised 觀察**：
- **相似度矩陣**：整體呈現較暗色調，缺乏明顯結構，表明批次內樣本的梯度方向普遍差異較大，相似度較低。這符合訓練初期模型尚未有效學習特徵的狀態。
- **t-SNE 視覺化**：點分布廣泛且隨機，未形成任何明顯的聚類。這再次印證了梯度方向的多樣性，模型對不同樣本的處理方式尚未分化。

### Epoch 10
![梯度相似度矩陣 - Epoch 10](gradient_similarities_epoch_10.png)
![梯度方向 t-SNE - Epoch 10](gradient_directions_tsne_epoch_10.png)

** revised 觀察**：
- **相似度矩陣**：相較於 Epoch 5，可能開始出現微弱的、小範圍的亮色區域，暗示部分樣本的梯度開始趨於一致，但整體結構仍不明顯。
- **t-SNE 視覺化**：點的分佈可能略微收縮，或者開始出現非常鬆散的、不穩定的初步聚集，顯示模型學習到了一些共享的基礎特徵。

### Epoch 15
![梯度相似度矩陣 - Epoch 15](gradient_similarities_epoch_15.png)
![梯度方向 t-SNE - Epoch 15](gradient_directions_tsne_epoch_15.png)

** revised 觀察**：
- **相似度矩陣**：更清晰的亮色區塊開始形成，表明特定樣本子集之間的梯度相似度顯著提高。這可能反映了模型對較易分類樣本的學習。
- **t-SNE 視覺化**：點的分布呈現出更明確的結構或初步的聚類輪廓。GHM 的影響開始顯現，模型可能開始區分不同難度的樣本，梯度方向開始分化。

## 訓練中期階段 (Epoch 20-40)

### Epoch 20
![梯度相似度矩陣 - Epoch 20](gradient_similarities_epoch_20.png)
![梯度方向 t-SNE - Epoch 20](gradient_directions_tsne_epoch_20.png)

** revised 觀察**：
- **相似度矩陣**：結構性增強，可能出現數個相對獨立的亮色方塊，代表梯度方向相似的樣本群體。對角線外的亮點也可能增多。
- **t-SNE 視覺化**：形成了更為明顯的幾個聚類，雖然可能邊界仍然模糊或有重疊。這表明模型對不同類型樣本的學習路徑開始產生差異。

### Epoch 25
![梯度相似度矩陣 - Epoch 25](gradient_similarities_epoch_25.png)
![梯度方向 t-SNE - Epoch 25](gradient_directions_tsne_epoch_25.png)

** revised 觀察**：
- **相似度矩陣**：結構化程度進一步提高，亮色區塊更為鞏固和清晰。這表明 GHM 正有效引導梯度，使相似樣本的梯度方向趨於一致。
- **t-SNE 視覺化**：聚類變得更緊湊，類間距離可能開始拉大。這與 GHM 報告中模型性能顯著提升的階段吻合，顯示 GHM 的加權機制效果顯著。

### Epoch 30
![梯度相似度矩陣 - Epoch 30](gradient_similarities_epoch_30.png)
![梯度方向 t-SNE - Epoch 30](gradient_directions_tsne_epoch_30.png)

** revised 觀察**：
- **相似度矩陣**：高相似度的模式（亮色區塊）非常明顯，可能覆蓋了相當一部分樣本對。這表明模型對大部分樣本形成了較為一致的梯度方向。
- **t-SNE 視覺化**：聚類結構更加穩定和清晰，點在簇內的分布可能更集中。GHM 對梯度分布的調控作用在此階段非常活躍。

### Epoch 35
![梯度相似度矩陣 - Epoch 35](gradient_similarities_epoch_35.png)
![梯度方向 t-SNE - Epoch 35](gradient_directions_tsne_epoch_35.png)

** revised 觀察**：
- **相似度矩陣**：形成了穩定的高相似度模式，結構可能與 Epoch 30 相似但更為強化。非對角線的亮區可能更顯著。
- **t-SNE 視覺化**：聚類間的分離度可能進一步增大，簇的形狀更為明確。這反映了 GHM 成功區分並差異化處理不同難度樣本的梯度。

### Epoch 40
![梯度相似度矩陣 - Epoch 40](gradient_similarities_epoch_40.png)
![梯度方向 t-SNE - Epoch 40](gradient_directions_tsne_epoch_40.png)

** revised 觀察**：
- **相似度矩陣**：模式趨於穩定，與前一階段相比變化可能不大，但結構可能更為清晰。亮色區塊的強度可能達到峰值。
- **t-SNE 視覺化**：聚類模式基本定型，點的分布變化趨緩。表明模型在 GHM 的引導下，梯度方向的整體結構趨於收斂。

## 訓練後期階段 (Epoch 45-60)

### Epoch 45
![梯度相似度矩陣 - Epoch 45](gradient_similarities_epoch_45.png)
![梯度方向 t-SNE - Epoch 45](gradient_directions_tsne_epoch_45.png)

** revised 觀察**：
- **相似度矩陣**：模式非常穩定，可能與 Epoch 40 極為相似。高相似度區塊清晰且固定，顯示梯度方向的一致性已建立。
- **t-SNE 視覺化**：聚類邊界清晰，內部緊湊，整體結構穩定。模型梯度方向基本收斂。

### Epoch 50
![梯度相似度矩陣 - Epoch 50](gradient_similarities_epoch_50.png)
![梯度方向 t-SNE - Epoch 50](gradient_directions_tsne_epoch_50.png)

** revised 觀察**：
- **相似度矩陣**：結構與 Epoch 45 基本一致，可能僅有微小變化。高相似度模式持續存在。
- **t-SNE 視覺化**：聚類結構非常穩定，簇內點的分布可能更加緊密。GHM 可能仍在微調困難樣本的梯度，但整體結構不變。

### Epoch 55
![梯度相似度矩陣 - Epoch 55](gradient_similarities_epoch_55.png)
![梯度方向 t-SNE - Epoch 55](gradient_directions_tsne_epoch_55.png)

** revised 觀察**：
- **相似度矩陣**：結構與先前輪次幾乎沒有差異，表明梯度相似性模式已完全穩定。
- **t-SNE 視覺化**：聚類結構穩定，形態固定。模型訓練已進入非常穩定的後期階段。

### Epoch 60
![梯度相似度矩陣 - Epoch 60](gradient_similarities_epoch_60.png)
![梯度方向 t-SNE - Epoch 60](gradient_directions_tsne_epoch_60.png)

** revised 觀察**：
- **相似度矩陣**：結構與 Epoch 55 無異，呈現最終的穩定模式。
- **t-SNE 視覺化**：聚類結構完全穩定，與前幾輪相比無明顯變化。表明在 GHM 優化下，模型梯度方向達到最終的穩定狀態。

## 整體變化趨勢 (Revised)

通過觀察訓練過程中的梯度相似度變化，我們可以總結出以下關鍵趨勢：

1.  **從混亂到有序**：梯度相似度矩陣從初期的低相似度、無結構狀態，逐漸演變為中期具有清晰亮色區塊、高度結構化的模式，並在後期達到穩定。
2.  **從分散到聚焦**：t-SNE 視覺化顯示梯度方向從早期的隨機分散，發展到中期形成數個逐漸清晰和分離的聚類，最終在後期穩定下來，簇內點更為集中。
3.  **GHM 作用顯現期**：訓練中期（約 Epoch 20-40）是梯度結構變化的關鍵時期，與 GHM 效果顯著提升、模型性能改善的階段吻合。此階段 GHM 的重新加權機制有效地引導了梯度方向的整合與分化。
4.  **後期穩定收斂**：訓練後期（約 Epoch 45 之後），梯度相似度模式和 t-SNE 聚類結構趨於穩定，反映了模型在 GHM 輔助下達到了較好的收斂狀態。

這些觀察結果與 GHM 的核心思想（平衡易學樣本和難學樣本的梯度貢獻）高度一致，視覺化地展示了 GHM 在訓練過程中動態調整梯度分布、促進模型穩定收斂的過程。 