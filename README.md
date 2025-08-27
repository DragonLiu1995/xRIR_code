<h1 align="center">Hearing Anywhere in Any Environment</h1> 
<h4 align="center" style="color:gray">
  <a href="https://dragonliu1995.github.io/" target="_blank">Xiulong Liu</a>,
  <a href="https://anuragkr90.github.io/" target="_blank">Anurag Kumar</a>,
  <a href="https://www.linkedin.com/in/paul-calamia/" target="_blank"> Paul Calamia</a>,
  <a href="https://scholar.google.com/citations?user=UyCazCsAAAAJ&hl=en" target="_blank"> Sebasti`a V. Amengual</a>,
  <a href="https://www.linkedin.com/in/calvinmurdock/" target="_blank"> Calvin Murdock </a>,
  <a href="https://www.ishwarya.me/" target="_blank"> Ishwarya Ananthabhotla </a>,
  <a href="https://www.linkedin.com/in/philrob22/" target="_blank"> Philip Robinson </a>,
  <a href="https://faculty.washington.edu/shlizee/NW/index.html" target="_blank"> Eli Shlizerman </a>,
  <a href="https://www.vamsiithapu.com/" target="_blank"> Vamsi Krishna Ithapu </a>,
  <a href="https://ruohangao.github.io/" target="_blank"> Ruohan Gao </a>
</h4>

<h4 align="center" style="color:gray">
  University of Washington, University of Maryland College Park, Meta Reality Labs Research
</h4>
<h5 align="center"> (If you find this project helpful, please give us a star ⭐ on this GitHub repository to support us.) </h5>

## 📋 Table of Contents
- [📋 Table of Contents](#-table-of-contents)
- [📝 Overview](#-overview)
- [🛠️ Installation](#️-installation)
  - [1. Clone the repository and create environment](#1-clone-the-repository-and-create-environment)
  - [2. Install extra dependencies](#2-install-extra-dependencies)
- [📊 Dataset](#-dataset)
  - [1. Download "AcousticRooms" Dataset](#1-download-AcousticRooms-dataset)
- [✅ Evaluation](#-evaluation)
  - [🌎 Pretrained model](#-pretrained-model)
- [Sim to Real](#-sim-to-real)
- [📧 Contact](#-contact)
- [📑 Citation](#-citation)

## 📝 Overview

**xRIR** is a novel and generalizable framework for cross-room RIR prediction. The approach demonstrates strong performance not only on large-scale synthetic dataset but also achieves decent performance when adapted to real acoustic scenes. This repository contains the unofficial implementation of the CVPR 2025 paper.

## 🛠️ Installation

### 1. Clone the repository and create environment
Clone the repository and create a conda environment:
```bash
git clone https://github.com/DragonLiu1995/xRIR_code.git
conda create -n xRIR python=3.8
conda activate xRIR
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Install extra dependencies
Install dependencies: pip install -r requirements.txt

## 📊 Dataset

### 1. Download "AcousticRooms" Dataset

Check out the official dataset repository at for details: https://github.com/facebookresearch/AcousticRooms. Download all and unzip all *.zip files to a data folder.


## ✅ Evaluation

Here we provide checkpoints for xRIR under 8-shot scenario for both seen and unseen splits in AcousticRooms dataset. To evaluate the model:
1. ``` export PYTHONPATH=$PYTHONPATH:[repo_directory]```
2. Run:

```
python eval_unseen.py
```
for unseen test split, and

```bash
python eval_seen.py
```
for seen test split.

### 🌎 Pretrained model 
Download our pretrained model checkpoints from [here](https://drive.google.com/file/d/12uL5u15gtRtiaLfCjPIN4nEDOVl4tdKi/view?usp=drive_link)

### Sim to Real
Check sim_to_real folder for more info. Basically sim to real transfer features two stages finetuning to achieve the most optimal results: 1. finetune on training split of 3 different rooms (12 * 3 as illustrated in this code, dampened room excluded), and then in second stage, we tune specifically on targeted room only (using only 12 samples in that room based on stage 1 checkpoint). For reference RIR, we preprocessed all RIRs in a room by dividing the waveform by training set's largest magnitude (12 samples in that room). We provide the rendered source depth map at the source in each room under `sim_to_real/depth_map`. For all other inputs including raw RIRs, xyz locations, you can obtain from original HearingAnythingAnywhere dataset.

## 📧 Contact

If you have any questions or need further assistance, feel free to reach out to us:

- Xiulong Liu: liuxiulong1995@gmail.com

## 📑 Citation
If you use this code for your research, please cite our work:
```
@inproceedings{liu2025hearing,
  title={Hearing Anywhere in Any Environment},
  author={Liu, Xiulong and Kumar, Anurag and Calamia, Paul and Amengual, Sebastia V and Murdock, Calvin and Ananthabhotla, Ishwarya and Robinson, Philip and Shlizerman, Eli and Ithapu, Vamsi Krishna and Gao, Ruohan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5732--5741},
  year={2025}
}
```
