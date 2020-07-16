# Prototype Mixture Models
Anonymous code for ECCV20 "Prototype Mixture Models for Few-shot Semantic Segmentation", paper id 700.

PMMs architecture:
![](img/PMMs.jpg)
RPMMS architecture:
![](img/RPMMs.jpg)

## Overview
This code can test the RPMMs and PMMs on Pascal voc dataset.
- `data/` contains the dataloader and dataset for inference;
- `config/` contains the config file;
- `networks/` contains the implementation of the PMMs(`FPMMs.py`) & RPMMs(`FRPMMs.py`);
- `models/` contains the backbone & PMMs module;
- `snapshots/` contains the FRPMMs parameters;
- `utils/` contains other dependent code :

## Dependencies
python == 3.7,
pytorch1.0,

torchvision,
pillow,
opencv-python,
pandas,
matplotlib,
scikit-image

## Inference
Note that you should modify the data path and model path in `config/settings.py` & `data/voc_val.py`.
```
python test_frame.py
```
##Citations

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{PMMs2020,
  title   =  {Prototype Mixture Models for Few-shot Semantic Segmentation},
  author  =  {Boyu Yang and Chang Liu and Bohao Li and Jianbin Jiao, and Ye, Qixiang},
  booktitle =  {ECCV},
  year    =  {2020}
}
```
