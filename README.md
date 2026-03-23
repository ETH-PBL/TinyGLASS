# TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection**

Pietro Bonazzi, Rafael Sutter, Luigi Capogrosso, Mischa Buob, Michele Magno

[ArXiv Preprint Link](https://arxiv.org/abs/2603.16451) &
[Dataset Link](https://zenodo.org/records/19186667)

## Table of Contents
* [📖 Introduction](#introduction)
* [🔧 Environments](#environments)
* [📊 Data Preparation](#data-preparation)
* [🚀 Run Experiments](#run-experiments)
* [📂 Dataset Release](#dataset-release)
* [🔗 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)
* [📜 License](#license)

## Introduction
This repository contains source code for TinyGLASS implemented with PyTorch.
TinyGLASS, a lightweight adaptation of the GLASS framework designed for real-time in-sensor anomaly detection on the Sony IMX500.

## Environments
Create a new conda environment and install required packages.
```
conda create -n glass_env python=3.9.15
conda activate glass_env
pip install -r requirements.txt
```
Experiments are conducted on 3x NVIDIA A6000 (48GB).
Same GPU and package version are recommended. 

## Data Preparation
The public datasets employed in the paper are listed below.

- MMS ([Download link](https://zenodo.org/records/19186667))
- MVTec AD ([Download link](https://www.mvtec.com/company/research/datasets/mvtec-ad/))

Other valuable datasets: 

- VisA ([Download link](https://github.com/amazon-science/spot-diff/))
- MPDD ([Download link](https://github.com/stepanje/MPDD/))

## Run Experiments
To reproduce the results in the paper please run :

```
bash run_all.sh
```

## Dataset Release

### 1.MMS Dataset ([Download link](https://zenodo.org/records/19186667))
The MMS Dataset comprises four defect categories for M&Ms candies, i.e., crack-hole, scratch, half, and normal, covering structural and surface-level anomalies. It was collected using an high-resolution microscope camera (mms_stretch) and the IMX500 camera (mms_rpi) 
![](figures/MMS_samples.png)


## Citation
Please cite the following paper if the code and dataset help your project:

```bibtex
@misc{bonazzi2026tinyglassrealtimeselfsupervisedinsensor,
      title={TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection}, 
      author={Pietro Bonazzi and Rafael Sutter and Luigi Capogrosso and Mischa Buob and Michele Magno},
      year={2026},
      eprint={2603.16451},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16451}, 
}
```

## Acknowledgements
Thanks for the great inspiration from [GLASS](https://github.com/cqylunlun/GLASS).

## License
The code and dataset in this repository are licensed under the [MIT license](https://mit-license.org/).
