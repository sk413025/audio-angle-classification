# GHM Training Analysis Report

## Training Parameters

- **Material:** plastic
- **Frequency:** 3000hz
- **Timestamp:** 20250411_225345
- **Random Seed:** 42
- **GHM Bins:** 10
- **GHM Alpha:** 0.75
- **GHM Margin:** 10.0
- **Total Epochs:** 60

## Training and Validation Performance

![Training Performance](training_comparison_20250411_225345.png)

*Figure 1: Training and validation performance comparing original loss vs GHM loss*

### Best Model Performance

- **Best Validation Accuracy:** 87.05% (Epoch 12)
- **Corresponding Training Accuracy:** 67.65%
- **Original Validation Loss:** 5.4841
- **GHM Validation Loss:** 0.9484

## Gradient Distribution Analysis

The following plots show the gradient distribution at different stages of training.
This helps visualize how GHM loss reshapes the gradient to focus on more informative samples.

### Gradient Evolution During Training

![Epoch 1 Gradient Distribution](gradient_plots/epoch1_batch0_20250411_225345.png)

*Figure 2: Gradient distribution at Epoch 1*

![Epoch 31 Gradient Distribution](gradient_plots/epoch31_batch0_20250411_225345.png)

*Figure 3: Gradient distribution at Epoch 31*

![Epoch 60 Gradient Distribution](gradient_plots/epoch60_batch0_20250411_225345.png)

*Figure 4: Gradient distribution at Epoch 60*

## GHM Statistics Analysis

![Epoch 1 GHM Stats](stats_analysis/ghm_stats_epoch_1.png)

*Figure 5: GHM gradient bin distribution at Epoch 1*

**GHM Statistics for Epoch 1:**

- Total gradients: 15
- Mean gradient: 0.7247
- Median gradient: 1.0000
- Min gradient: 0.0000
- Max gradient: 2.0000
- Std deviation of gradients: 0.7542
- Most populated bin: 0 (93.33%)
- Least populated bin: 2 (0.00%)
- Gradient distribution evenness: 27.85% (standard deviation)

![Epoch 31 GHM Stats](stats_analysis/ghm_stats_epoch_31.png)

*Figure 6: GHM gradient bin distribution at Epoch 31*

**GHM Statistics for Epoch 31:**

- Total gradients: 14
- Mean gradient: 0.7791
- Median gradient: 1.0000
- Min gradient: 0.0000
- Max gradient: 2.0000
- Std deviation of gradients: 0.7767
- Most populated bin: 0 (100.00%)
- Least populated bin: 1 (0.00%)
- Gradient distribution evenness: 30.00% (standard deviation)

![Epoch 60 GHM Stats](stats_analysis/ghm_stats_epoch_60.png)

*Figure 7: GHM gradient bin distribution at Epoch 60*

**GHM Statistics for Epoch 60:**

- Total gradients: 14
- Mean gradient: 0.6563
- Median gradient: 1.0000
- Min gradient: 0.0000
- Max gradient: 2.0000
- Std deviation of gradients: 0.6427
- Most populated bin: 0 (100.00%)
- Least populated bin: 1 (0.00%)
- Gradient distribution evenness: 30.00% (standard deviation)

## Conclusion

### Overall Training Results

- **Initial Validation Accuracy:** 63.84%
- **Final Validation Accuracy:** 82.14%
- **Overall Improvement:** 18.30%

The model shows positive improvement during training with GHM loss.

**Loss Analysis:**

- Original loss reduction: 1.8935
- GHM loss reduction: 0.0271

### Effectiveness of GHM

The Gradient Harmonizing Mechanism (GHM) was applied to balance the training process by reshaping the gradient distributions. Based on the visualizations and statistics, we can observe how the gradient distribution evolved throughout training.

GHM helps balance the contribution of easy vs. hard examples, potentially leading to more robust model training especially in cases where the dataset contains imbalanced examples or difficulty levels.

The final model checkpoint can be found at: 
`./saved_models/model_checkpoints/plastic_3000hz_ghm/model_ghm_epoch_60_20250411_225345.pt`
