# TCWD   

Paper: TCWD: Temporal compressive coherent diffraction imaging enhanced by weighted wavelet domain diffusion model   

Authors: Dingxiang Yuanⁱᵃ, Rundong Gongⁱᵇ, Yaolong Faᶜ, Xianghong Zouᵈ, Tianshui Yuᵉ, Wenbo Wanᵃᶠ, Qiegen Liuᵃᶠ   

ⁱThese authors contributed equally to this work   
ᵃSchool of Information Engineering, Nanchang University, Nanchang 330031, China   
ᵇSchool of Mathematics and Computer Sciences, Nanchang University, Nanchang 330031, China   
ᶜJi Luan Academy, Nanchang University, Nanchang 330031, China   
ᵈSchool of Advanced Manufactory, Nanchang University, Nanchang 330031, China   
ᵉNanchang Yijing Information Technology Co., Ltd, Nanchang 330108, China   
ᶠJiangxi Provincial Key Laboratory of Advanced Signal Processing and Intelligent Communications, Nanchang University, Nanchang 330031, China   

Journal: Optics Communications 597 (2025) 132587   
DOI: https://doi.org/10.1016/j.optcom.2025.132587   
Corrigendum: https://doi.org/10.1016/j.optcom.2025.132703   
Date: October 24, 2025 (Accepted)   

## Abstract

Temporal compressive coherent diffraction imaging (TC-CDI) is a lensless imaging technique that enables dynamic process visualization by compressive sampling in the frequency domain. While diffusion models improve reconstruction quality, they struggle with complex targets. We propose TCWD, which integrates a weighted wavelet domain diffusion model to balance high- and low-frequency components, and a spatial domain diffusion model to reduce distortions. These dual-domain learned priors guide the iterative hybrid input-output (HIO) algorithm, enhancing reconstruction accuracy while maintaining interpretability. Validated on Fashion MNIST, Quick Draw datasets, and measured data, TCWD achieves superior detail recovery, artifact suppression, and structural integrity with an average PSNR of 28.14 dB and SSIM of 0.970.   

## Keywords

Coherent diffraction imaging, Prior learning, Weighted wavelet domain, Image reconstruction, Generative diffusion model   
   
<img width="1024" height="665" alt="laser" src="https://github.com/user-attachments/assets/458092bc-e085-4b69-8d7c-c90a32750827" />

## Key Contributions

- Proposes a cross-domain prior extraction strategy combining Fourier transform, wavelet decomposition, and weighting, enhancing data sparsity and feature extraction in the frequency domain.
<img width="13788" height="3171" alt="jiaquan" src="https://github.com/user-attachments/assets/8f40adec-8682-41cd-bdf3-2960940a9c0d" />
- Develops a joint prior-constrained reconstruction method that recovers sparsely sampled spectral information and high-quality spatial-domain images.   
- Outperforms state-of-the-art methods (TC-CDI, FDTC) in handling complex dynamic targets, with robust performance across simulated and real-world experimental data.   

## Method Overview

TCWD consists of three core stages:
1. **Prior Learning (Weighted Wavelet Domain)**: Transforms spatial data to frequency domain via Fourier transform, decomposes into 8 sub-bands with DWT, and applies weighting to balance frequency components. Trains a VE-SDE based diffusion model to learn frequency-domain priors.
<img width="1806" height="594" alt="27cfea41e24c07c87aa1c73a2bf775dc" src="https://github.com/user-attachments/assets/fda6ecb8-69b1-4614-9d29-3418dfb29a6b" />
2. **Prior Learning (Spatial Domain)**: Trains a complementary diffusion model to preserve structural integrity and suppress artifacts in the spatial domain.
3. **Iterative Reconstruction**: Unfolds snapshot measurements into multi-frame frequency-domain data, integrates dual-domain priors into HIO-based phase retrieval, and applies real-space correction for stability.
<img width="10642" height="12333" alt="流程图定稿4 15改(1)" src="https://github.com/user-attachments/assets/e4da04bb-0db5-44f9-a7e4-4bfcd01ab8b9" />

## Requirements and Dependencies

Python==3.8   
Pytorch==1.7.0   
tensorflow==2.4.0   
torchvision==0.8.0   
tensorboard==2.7.0   
scipy==1.7.3   
numpy==1.19.5   
ninja==1.10.2   
matplotlib==3.5.1   
jax==0.2.26   

## Checkpoints

Pretrained models are available for download:
- **Time-domain Unfolding Model**: [Baidu Cloud](https://pan.baidu.com/s/13dLPw9DeBt5cwrmuJ8MEUg?pwd=ydxm) (Password: ydxm)
  Extract `Time_domain_unfolding` zip file and place in `TCWD/Time_domain_unfolding/model/base`
- **Dual-domain Generation Models**: [Baidu Cloud](https://pan.baidu.com/s/13dLPw9DeBt5cwrmuJ8MEUg?pwd=ydxm) (Password: ydxm)
  Extract `Spatial` and `Wavelet` zip files for spatial-domain and weighted wavelet-domain pre-trained models

## Dataset

### Training/Validation Data
- [Baidu Cloud](https://pan.baidu.com/s/1p5LmDDs9nbikHIwaneBUyA?pwd=ydxx) (Password: ydxx)
- Extract `MNIST_512` zip file:
  - Use first 5,000 images in `Train-images` as training set (resized to 40×40 pixels, placed on 512×512 black background)
  - Use `images` folder for validation

### Test Datasets
- Quick Draw: 20 randomly selected images (resized to 40×40 pixels, 20 dynamic frames)
<img width="7183" height="913" alt="a1" src="https://github.com/user-attachments/assets/38b5c11f-23e7-4876-852e-a3d46cbdd478" />
- Fashion MNIST: 20 randomly selected grayscale images (resized to 40×40 pixels)
- Measured Data: Dynamic "westlake" target (minimum line width 20 μm, 780 nm laser source)

## Training Configuration

### Hyperparameters

|Parameter|Value|Description|
|---|---|---|
|Gaussian Noise (Wavelet Domain)|0.01–380 (std)|Noise range for weighted wavelet domain model training|
|Gaussian Noise (Spatial Domain)|0.01–4 (std)|Noise range for spatial domain model training|
|Training Iterations (Spatial)|300,000|Total iterations for spatial domain diffusion model|
|Training Iterations (Wavelet)|600,000|Total iterations for weighted wavelet domain model|
|Optimizer|Adam (β₁=0.9, β₂=0.999)|Optimization algorithm|
|Momentum|0.999|Exponential moving average for parameters|
|Maximum Gradient Magnitude|1.0|Gradient clipping threshold|
|Signal-to-Noise Ratio (SNR)|0.075|Dimensionless ratio for reverse diffusion|

### 1. Weighted Wavelet Domain Model Training
1. Modify training dataset path in `Code/TCWD/Iterative_Reconstruction/datasets_wavelet.py`
2. Run training command in `Code/TCWD/Iterative_Reconstruction` directory:

```bash

python main_wavelet.py --config=configs/ve/SIAT_kdata_wavelet_ncsnpp.py --workdir=exp_wavelet --mode=train --eval_folder=result
```

### 2. Spatial Domain Model Training
1. Modify training dataset path in `Code/TCWD/Iterative_Reconstruction/datasets_spatial.py`
2. Run training command in `Code/TCWD/Iterative_Reconstruction` directory:

```bash

python main_spatial.py --config=configs/ve/SIAT_kdata_spatial_ncsnpp.py --workdir=exp_spatial --mode=train --eval_folder=result
```

## Iterative Reconstruction

### Step 1: Time Domain Unfolding
Run in `Code/TCWD/Time_domain_compression` directory to decompress single snapshot into multiple frames:

```bash

python test.py
```

### Step 2: Reconstruction Configuration
Modify model and test dataset paths in `Code/TCWD/Iterative_Reconstruction/PCsampling_demo.py`

**Key reconstruction parameters**:
- Number of SDE discretization steps (N): 600
- Number of dynamic frames (K): 20
- Spatial domain weighting parameters: a=0.5, b=0.8

### Step 3: Run Reconstruction
Execute in `Code/TCWD/Iterative_Reconstruction` directory:

```bash

python PCsampling_demo.py
```

## Experimental Results

### Quantitative Performance (Average Values)

|Dataset|Method|PSNR (dB)|SSIM|
|---|---|---|---|
|Quick Draw|TC-CDI|25.38|0.950|
|Quick Draw|FDTC|26.43|0.956|
|Quick Draw|TCWD|28.14|0.970|
<img width="9192" height="7438" alt="Fig  6" src="https://github.com/user-attachments/assets/fa123767-598f-40d4-b3f2-9d5b55ffd4f9" />

### Key Highlights
- Maximum PSNR: 34.96 dB (Quick Draw dataset)
- Maximum SSIM: 0.992 (Quick Draw dataset)
<img width="9192" height="7446" alt="2" src="https://github.com/user-attachments/assets/f1d80bff-c865-4105-ab2e-843dac918dc3" />
- Superior artifact suppression and detail preservation on complex dynamic targets
- Stable performance on measured data (complete "westlake" target reconstruction)
<img width="9125" height="4792" alt="4" src="https://github.com/user-attachments/assets/1a3792a5-3804-4b79-b68b-d868f3824a5c" />

## Ablation Study

|Method Configuration|Quick Draw (PSNR/SSIM)|Fashion MNIST (PSNR/SSIM)|
|---|---|---|
|TC-CDI (No Priors)|23.00/0.936|24.88/0.931|
|TC-CDI + Spatial Priors|25.52/0.964|24.75/0.927|
|FDTC (Frequency Priors)|28.15/0.974|26.25/0.946|
|TCWD (Wavelet Priors Only)|30.12/0.977|26.68/0.951|
|TCWD (Dual-Domain Priors)|31.13/0.985|27.45/0.952|
![Uploading 5.png…]()

## Target Size Analysis
TCWD maintains superior performance across different target sizes:
- 30×30 pixels: Best structural integrity and lowest noise
- 60×60 pixels: Outperforms TC-CDI/FDTC by 3.58 dB (PSNR) and 0.045 (SSIM)
- Scalable to large complex targets with minimal quality degradation
<img width="7538" height="8300" alt="Fig  11" src="https://github.com/user-attachments/assets/fc21006b-c7b9-4d8b-b847-6fcc4bf2e579" />

## Funding
This study was supported by:
- National Natural Science Foundation of China (No. U24A20304)
- Jiangxi Provincial Natural Science Foundation (Nos. 20242BAB20040, 20252BAC240032, 20242BCC32016)
