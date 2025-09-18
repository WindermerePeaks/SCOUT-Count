# SCOUT-Count: PyTorch Implementation

Official PyTorch implementation of the paper **"SCOUT-Count: Stylization with Covariance-Whitening and Uncertainty for Domain-Generalized Remote Sensing Object Counting"**.

## Environment Setup

The following environment setup was used to ensure reproducibility:
- Python 3.8
- CUDA Toolkit 11.3.1
- PyTorch 1.11.0
- NumPy 1.23.0
- Matplotlib 3.6.2
- Pandas 2.0.3
- Pillow 9.4.0

Ensure these dependencies are installed using the following command:

```bash
pip install -r requirements.txt
```

## Usage


### Test
We provide a method to test on public/custom datasets. The dataset should be organized in the following structure:

```
└── datasets
    └── dataset_name
        └── test
            ├── den
            │   ├── 1.npy
            │   ├── 2.npy
            │   └── ...
            ├── img
            │   ├── 1.jpg
            │   ├── 2.jpg
            │   └── ...
            └── cam
                ├── 1.npy
                ├── 2.npy
                └── ...
```

Once your dataset is properly structured, you can run the following command to test:

```bash
python test.py
```

## Pre-trained Models

Coming soon.

## Pre-trained VLMs

This project leverages **CLIP-RS**, a pre-trained vision-language model tailored for remote sensing.  
The model was introduced in the following paper:

> Shi, H., Tan, Z., Zhang, Z., Wei, H., Hu, Y., Zhang, Y., Chen, Z., 2025.  *Remote sensing semantic segmentation quality assessment based on vision language model.*  arXiv:2502.13990.

