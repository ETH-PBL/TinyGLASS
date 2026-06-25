# TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection

Pietro Bonazzi, Rafael Sutter, Luigi Capogrosso, Mischa Buob, Michele Magno

[ArXiv](https://arxiv.org/abs/2603.16451) &
[Dataset](https://zenodo.org/records/19186667) &
[Checkpoints](https://huggingface.co/pietrobonazzi/TinyGLASS)

## Table of Contents
* [📖 Introduction](#introduction)
* [🔧 Setup](#setup)
* [📊 Data Preparation](#data-preparation)
* [🚀 Training](#training)
* [📦 Pretrained Checkpoints](#pretrained-checkpoints)
* [🎥 IMX500 Deployment](#imx500-deployment)
* [📂 Dataset Release](#dataset-release)
* [🔗 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)
* [📜 License](#license)

## Introduction
TinyGLASS is a lightweight adaptation of the GLASS framework for real-time in-sensor anomaly detection on the Sony IMX500.

## Setup
Install dependencies with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```
Experiments are conducted on 3× NVIDIA A6000 (48 GB).

## Data Preparation
Download the datasets:
```bash
# MVTec AD
# Download from https://www.mvtec.com/company/research/datasets/mvtec-ad
# Extract to /datasets/pbonazzi/tinyglass_mvtec/

# MMS (M&Ms candies)
wget "https://zenodo.org/records/19186667/files/tinyglass_mmdataset.zip?download=1" -O tinyglass_mmdataset.zip
unzip tinyglass_mmdataset.zip -d /datasets/pbonazzi/

# DTD textures (augmentation)
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz -C /datasets/pbonazzi/tinyglass_mvtec/
```

Expected layout:
```
/datasets/pbonazzi/
├── tinyglass_mvtec/
│   ├── bottle/ carpet/ ...   # 15 MVTec classes
│   └── dtd/images/           # DTD augmentation textures
└── tinyglass_mmdataset/
    ├── mms_rpi/              # IMX500 camera captures
    └── mms_stretch/          # Microscope camera captures
```

Other datasets used in the paper:
- VisA ([link](https://github.com/amazon-science/spot-diff/))
- MPDD ([link](https://github.com/stepanje/MPDD/))

## Training
```bash
# All datasets across 3 GPUs
bash shell/run_all.sh

# Individual datasets
bash shell/run_tinyglass_mvtec_gpu0.sh   # carpet grid leather tile wood bottle cable capsule
bash shell/run_tinyglass_mvtec_gpu1.sh   # hazelnut metal_nut pill screw toothbrush transistor zipper
bash shell/run_tinyglass_mms.sh          # mms_rpi
```
Checkpoints are saved to `results/<dataset>/models/backbone_0/<class>/ckpt_best_<epoch>.pth`.

## Pretrained Checkpoints
Download from [HuggingFace](https://huggingface.co/pietrobonazzi/TinyGLASS):
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="pietrobonazzi/TinyGLASS", local_dir="checkpoints")
```

| Dataset | Class | Best epoch | Image AUROC |
|---------|-------|-----------|------------|
| MMS | mms_rpi | in progress | in progress |
| MVTec | carpet | in progress | in progress |
| MVTec | hazelnut | in progress | in progress |
| MVTec | … | in progress | in progress |

## IMX500 Deployment

The full pipeline to deploy a trained checkpoint to the Sony IMX500:

```
ckpt_best_*.pth  →  [pth2onnx]  →  .onnx  →  [MCT quantize]  →  qmodel.onnx  →  [imx500-converter]  →  network.rpk
```

### Step 1 — Export to ONNX
```bash
uv run python onnx/pth2onnx2.py \
    --model_dir mvtec_mms_rpi \
    --backbone resnet18 \
    --layers layer2 layer3 \
    --input_size 3 256 256 \
    --pre_dim 384 \
    --tgt_dim 384
# Output: results/tinyglass_mms/models/backbone_0/mvtec_mms_rpi/mvtec_mms_rpi_simplified.onnx
```

### Step 2 — Post-Training Quantization (MCT)
Requires [Model Compression Toolkit](https://github.com/sony/model_optimization) and the test dataset for calibration:
```bash
uv run python mct_pt_full.py \
    --model_dir mvtec_mms_rpi \
    --data_path /datasets/pbonazzi/tinyglass_mmdataset \
    --aug_path /datasets/pbonazzi/tinyglass_mvtec/dtd/images \
    --subdatasets mms_rpi \
    --input_size 3 256 256 \
    --pre_dim 384 \
    --tgt_dim 384
# Output: results/.../mvtec_mms_rpi/qmodel.onnx
```

### Step 3 — Convert to IMX500 RPK
Install the [imx500-converter](https://developer.sony.com/imx500) from Sony and run:
```bash
imx500-converter \
    --onnx results/tinyglass_mms/models/backbone_0/mvtec_mms_rpi/qmodel.onnx \
    --output-dir results/tinyglass_mms/models/backbone_0/mvtec_mms_rpi/converted/
# Output: .../converted/network.rpk
```

### Step 4 — Run on Device

**Image inference** (any machine):
```bash
uv run python demo_imx500.py image \
    --checkpoint checkpoints/models/backbone_0/mvtec_mms_rpi/ckpt_best_7.pth \
    --input "path/to/images/*.png" \
    --output-dir results/demo/ \
    --threshold 0.1
```

**Live camera** (Raspberry Pi + IMX500):
```bash
python demo_imx500.py live \
    --rpk results/tinyglass_mms/models/backbone_0/mvtec_mms_rpi/converted/network.rpk \
    --checkpoint checkpoints/models/backbone_0/mvtec_mms_rpi/ckpt_best_7.pth \
    --threshold 0.1
```

The live view shows three panels side by side: raw frame | CPU GLASS heatmap | IMX500 NPU heatmap. Press `q` to exit.

## Dataset Release

### MMS Dataset ([Download](https://zenodo.org/records/19186667))
The MMS Dataset comprises four defect categories for M&Ms candies — crack-hole, scratch, half, and normal — covering structural and surface-level anomalies. Collected with a high-resolution microscope camera (`mms_stretch`) and the IMX500 camera (`mms_rpi`).

![](figures/MMS_samples.png)

## Citation
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
