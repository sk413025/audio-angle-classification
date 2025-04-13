# audio-angle-classification

A deep learning system for audio angle classification using spectrograms with multiple neural network architectures (CNN, ResNet, ConvNeXt)

## Project Introduction

The Audio Angle Classification project aims to accurately predict the angle of sound sources through spectogram analysis. This system is particularly useful in scenarios requiring precise sound direction identification, such as acoustic localization, smart speaker systems, and robotic auditory perception.

### Core Technologies

- **Audio Spectrogram Analysis**: Converts raw audio into spectrograms for deep learning processing
- **Ranking Learning**: Uses pairwise comparison methods to learn angle ranking relationships
- **Gradient Harmonizing Mechanism (GHM)**: Special loss function that more effectively handles difficult samples
- **Visualization Analysis Tools**: Provides visualization of various metrics during training, especially dynamic changes in GHM statistics

### Key Features

- Support for multiple network architectures (lightweight CNN, ResNet variants)
- Processing of audio data with different materials and frequencies
- Implementation of the innovative GHM loss function, improving resistance to sample imbalance
- Rich visualization and analysis tools

## Installation Guide

### Dependencies

This project requires the following main dependencies:

```
torch>=1.8.0
numpy>=1.19.0
matplotlib>=3.3.0
```

### Installation Steps

1. Clone this repository:

```bash
git clone https://github.com/sk413025/audio-angle-classification.git
cd audio-angle-classification
```

2. Create and activate a virtual environment (recommended):

```bash
conda create -n audio-class python=3.8
conda activate audio-class
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage Instructions

### Data Preparation

Place audio data in the `data` directory with the following structure:

```
data/
  ├── {class_name}/
  │     ├── {material}/
  │     │     ├── {material}_{class_name}_{frequency}_{seq_num}.wav
```

For example: `data/0_degree/plastic/plastic_0_degree_1000hz_1.wav`

### Model Training

#### Standard Training

Train with standard MarginRankingLoss:

```bash
python train.py --frequency 1000hz --material plastic --loss-type standard
```

#### GHM Training

Train with Gradient Harmonizing Mechanism loss:

```bash
python train.py --frequency 1000hz --material plastic --loss-type ghm --ghm-bins 10 --ghm-alpha 0.75
```

Parameter explanation:
- `--frequency`: Select frequency data to use (500hz, 1000hz, 3000hz, all)
- `--material`: Select material type
- `--loss-type`: Select loss function type (standard, ghm)
- `--ghm-bins`: Number of GHM bins
- `--ghm-alpha`: GHM alpha parameter

### Model Evaluation

Evaluate model performance:

```bash
python evaluate.py --model-path saved_models/model_checkpoints/plastic_1000hz_ghm_20250414_012345/model_epoch_30.pt
```

### Using Visualization Tools

#### GHM Visualization Analysis

Visualize single training statistics:

```python
from utils.visualization import visualize_ghm_stats

visualize_ghm_stats(
    "saved_models/stats/plastic_1000hz_ghm_20250414_011143/epoch30_batch0_20250414_011143.npy",
    output_path="visualization.png"
)
```

Compare GHM statistics across different epochs:

```python
from utils.visualization import compare_ghm_epochs

compare_ghm_epochs(
    "saved_models/stats/plastic_1000hz_ghm_20250414_011143",
    epochs=[1, 10, 20, 30],
    output_path="epoch_comparison.png"
)
```

Analyze the entire training process:

```python
from utils.visualization import analyze_ghm_training

analyze_ghm_training(
    "saved_models/stats/plastic_1000hz_ghm_20250414_011143",
    output_dir="analysis_output"
)
```

## Project Structure

```
audio-angle-classification/
  ├── train.py                 # Main training script
  ├── datasets.py              # Dataset definitions
  ├── config.py                # Configuration parameters
  ├── models/                  # Model definitions
  │     ├── resnet_ranker.py   # ResNet and CNN models
  ├── losses/                  # Loss functions
  │     ├── ghm_loss.py        # GHM loss function implementation
  ├── utils/                   # Tools and helper functions
  │     ├── common_utils.py    # Common utility functions
  │     ├── debugging_utils.py # Debugging tools
  │     ├── ghm_utils.py       # GHM-related tools
  │     ├── visualization/     # Visualization tools
  │           ├── ghm_analyzer.py     # GHM visualization analysis tools
  │           ├── plot_utils.py       # Chart plotting tools
  ├── saved_models/            # Saved models and statistics
```

### Core Modules

- **Model Module**: Defines neural network architectures for audio angle classification
- **Loss Function Module**: Implements standard and GHM loss functions
- **Dataset Module**: Handles loading and preprocessing of audio spectrogram data
- **Visualization Module**: Provides visualization tools for training and evaluation processes

### Visualization Module Features

The visualization module provides multiple functions to help understand the model training process:

- Training history visualization
- GHM bin statistics analysis
- Cross-epoch GHM variation comparison
- Gradient distribution visualization

## Models and Datasets

### Supported Model Architectures

- **SimpleCNNAudioRanker**: Lightweight CNN architecture, suitable for resource-constrained scenarios
- **ResNetAudioRanker** (planned): ResNet-based architecture for more complex scenarios

### Dataset Format

The project uses audio spectrogram datasets, supporting the following formats:

- Audio format: WAV files
- Frequency options: 500Hz, 1000Hz, 3000Hz
- Material options: plastic, metal, wood, etc.
- Angle classes: Multiple angle classes (e.g., 0_degree, 45_degree, etc.)

### Preprocessing Pipeline

1. Read WAV audio files
2. Convert to mono (if stereo)
3. Calculate Short-Time Fourier Transform (STFT)
4. Convert to spectrogram in decibel scale
5. Normalize

## Experimental Results

### Key Findings

- GHM loss function can effectively improve classification accuracy for difficult samples compared to standard loss functions
- Models trained with GHM perform more stably on imbalanced datasets
- 1000Hz frequency data performs best in most scenarios

### Performance Comparison

| Model           | Loss Function | Accuracy | Notes                         |
|-----------------|---------------|----------|-------------------------------|
| SimpleCNN       | Standard      | 85%      | Baseline model                |
| SimpleCNN       | GHM           | 91%      | Better on difficult samples   |

### Experiment Reports

For detailed experimental results, please refer to our reports:

- [GHM Training Analysis Report](experiments/20250411_GHM_Training_Analysis/README.md) - Gradient Harmonizing Mechanism training effect analysis (2025-04-11)

## Contributing

We welcome all forms of contributions, including but not limited to:

- Bug reports
- Feature requests
- Code improvements
- Documentation improvements

### Contribution Steps

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
