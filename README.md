# Prototype Mixture Models
ECCV20 "Prototype Mixture Models for Few-shot Semantic Segmentation"

PMMs architecture:
![PMMs](./img/PMMs.jpg)
RPMMS architecture:
![RPMMs](./img/RPMMs.jpg)

The training framework is coming soon！

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

## Training
```
python train.py
```

## Inference
Note that you should modify the data path and model path in `config/settings.py` & `data/voc_val.py`.
```
python test_frame.py
```
## Performance
| Backbone | Method | Pascal-5<sup>0</sup> | Pascal-5<sup>1</sup> | Pascal-5<sup>2</sup> | Pascal-5<sup>3</sup> | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| VGG16  | RPMMs | 47.14         | 65.82         | 50.57         | 48.54         | 53.02 |
| Resnet50|RPMMs|55.15|66.91|52.61|50.68|56.34

## Citations
Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{PMMs2020,
  title   =  {Prototype Mixture Models for Few-shot Semantic Segmentation},
  author  =  {Boyu Yang and Chang Liu and Bohao Li and Jianbin Jiao, and Ye, Qixiang},
  booktitle =  {ECCV},
  year    =  {2020}
}
```
