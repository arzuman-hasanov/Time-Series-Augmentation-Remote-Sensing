#Comparison of Time Series Augmentation Methods for Self-Supervised Contrastive Learning Applied to Remote Sensing Time Series

# List of implemented methods
| Method for Data Augmentation | Source |
|-----------------------------|--------|
| Jittering                   | --- |
|TADA                         |[Temporal Adversarial Data Augmentation for Time Series Data](https://arxiv.org/abs/2407.15174) |
| Rotation                    | https://arxiv.org/pdf/2206.13508 |
| Scaling                     | https://arxiv.org/pdf/2206.13508 |
| Flipping                    | --- |
| Resampling                  | --- |
| Permutation                 | https://arxiv.org/pdf/2206.13508 |
| Resizing                    | --- |
| WindowSlice,WindowWraping   |[Data Augmentation for Time Series Classification usingConvolutional Neural Networks](https://shs.hal.science/halshs-01357973/document)|
| MagnitudeWaripin,TimeWarping   |[Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks](https://arxiv.org/pdf/1706.00527)|
| PhasePerturbation           | (https://www.readcube.com/articles/10.3389/frai.2024.1414352) |
| MagnitudePerturbation       | (https://www.readcube.com/articles/10.3389/frai.2024.1414352) |
| Stiefelgen  |[StiefelGen: A Simple, Model Agnostic Approach for Time Series Data Augmentation over Riemannian Manifolds](https://arxiv.org/abs/2402.19287)|
| TemporalDropout  |[An Empirical Study on Data Augmentation for Pixelwise Satellite Image Time-Series Classification and Cross-Year Adaptation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10833777)|
|MixUp                        |  https://arxiv.org/pdf/2304.04271   |
|maskedtimeseries             |     |
| InterExtrapolation  |[Time-Series Data Augmentation based on Interpolation](https://www.sciencedirect.com/science/article/pii/S1877050920316914)|
| WhiteNoiseWindow  |[White Noise Windows: Data Augmentation for Time Series](https://www.researchgate.net/publication/352008668_White_Noise_Windows_Data_Augmentation_for_Time_Series)|
| CutMix  |[CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899)|




## Installation

### 1 - Environment Setup

1. **CPU**:
```bash
conda env create -f env.yml
conda activate timeseries_aug
```


### 2 - Data Setup

Install Hugging Face datasets:
```bash
pip install datasets
```


### 3 - Testing the Installation

CPU test:
```bash
python plot/main_plot.py augmentation=jittering   augmentations.jittering.sigma=10
python plot/views_plot.py augmentation=windowslice_v1 
```


