# GHM Training Analysis Report

## Training Parameters

- **Material:** plastic
- **Frequency:** 3000hz
- **Timestamp:** 20250411_222821
- **Random Seed:** 42
- **GHM Bins:** 10
- **GHM Alpha:** 0.75
- **GHM Margin:** 10.0
- **Total Epochs:** 60

## Training and Validation Performance

![Training Performance](training_comparison_20250411_222821.png)

*Figure 1: Training and validation performance comparing original loss vs GHM loss*

### Best Model Performance

- **Best Validation Accuracy:** 64.73% (Epoch 13)
- **Corresponding Training Accuracy:** 86.86%
- **Original Validation Loss:** 8.9071
- **GHM Validation Loss:** 1.8868

## Gradient Distribution Analysis

The following plots show the gradient distribution at different stages of training.
This helps visualize how GHM loss reshapes the gradient to focus on more informative samples.

### Gradient Evolution During Training

![Epoch 1 Gradient Distribution](gradient_plots/epoch1_batch0_20250411_222821.png)

*Figure 2: Gradient distribution at Epoch 1*

![Epoch 31 Gradient Distribution](gradient_plots/epoch31_batch0_20250411_222821.png)

*Figure 3: Gradient distribution at Epoch 31*

![Epoch 60 Gradient Distribution](gradient_plots/epoch60_batch0_20250411_222821.png)

*Figure 4: Gradient distribution at Epoch 60*

## GHM Statistics Analysis

![Epoch 1 GHM Stats](stats_analysis/ghm_stats_epoch_1.png)

*Figure 5: GHM gradient bin distribution at Epoch 1*

**GHM Statistics for Epoch 1:**

- Total gradients: 27
- Mean gradient: 0.3038
- Median gradient: 0.0000
- Min gradient: 0.0000
- Max gradient: 2.0000
- Std deviation of gradients: 0.6682
- Most populated bin: 0 (96.30%)
- Least populated bin: 1 (0.00%)
- Gradient distribution evenness: 28.79% (standard deviation)

![Epoch 31 GHM Stats](stats_analysis/ghm_stats_epoch_31.png)

*Figure 6: GHM gradient bin distribution at Epoch 31*

**GHM Statistics for Epoch 31:**

- Total gradients: 27
- Mean gradient: 0.2530
- Median gradient: 0.0000
- Min gradient: 0.0000
- Max gradient: 2.0000
- Std deviation of gradients: 0.5970
- Most populated bin: 0 (96.30%)
- Least populated bin: 1 (0.00%)
- Gradient distribution evenness: 28.79% (standard deviation)

![Epoch 60 GHM Stats](stats_analysis/ghm_stats_epoch_60.png)

*Figure 7: GHM gradient bin distribution at Epoch 60*

**GHM Statistics for Epoch 60:**

- Total gradients: 27
- Mean gradient: 0.2188
- Median gradient: 0.0000
- Min gradient: 0.0000
- Max gradient: 2.0000
- Std deviation of gradients: 0.5440
- Most populated bin: 0 (100.00%)
- Least populated bin: 1 (0.00%)
- Gradient distribution evenness: 30.00% (standard deviation)

## Conclusion

### Overall Training Results

- **Initial Validation Accuracy:** 61.16%
- **Final Validation Accuracy:** 62.95%
- **Overall Improvement:** 1.79%

The model shows positive improvement during training with GHM loss.

**Loss Analysis:**

- Original loss reduction: -0.4661
- GHM loss reduction: -0.5476

### Effectiveness of GHM

The Gradient Harmonizing Mechanism (GHM) was applied to balance the training process by reshaping the gradient distributions. Based on the visualizations and statistics, we can observe how the gradient distribution evolved throughout training.

GHM helps balance the contribution of easy vs. hard examples, potentially leading to more robust model training especially in cases where the dataset contains imbalanced examples or difficulty levels.

The final model checkpoint can be found at: 
`./saved_models/model_checkpoints/plastic_3000hz_ghm/model_ghm_epoch_60_20250411_222821.pt`
