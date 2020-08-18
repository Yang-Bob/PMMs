# Prototype Mixture Models
This code is for the paper "[Prototype Mixture Models for Few-shot Semantic Segmentation](https://arxiv.org/pdf/2008.03898.pdf)" in European Conference on Computer Vision(ECCV 2020).

PMMs architecture:
![PMMs](./img/PMMs.jpg)
RPMMS architecture:
![RPMMs](./img/RPMMs.jpg)


## Overview
This code can test the RPMMs and PMMs on Pascal voc dataset.
- `data/` contains the dataloader and dataset for inference;
- `config/` contains the config file;
- `networks/` contains the implementation of the PMMs(`FPMMs.py`) & RPMMs(`FRPMMs.py`);
- `models/` contains the backbone & PMMs module;
- `snapshots/` contains the FRPMMs parameters;
- `utils/` contains other dependent code;

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
<table>
    <tr>
        <td>Setting</td>
        <td>Backbone</td>
        <td>Method</td>
        <td>Pascal-5<sup>0</sup></td>
        <td>Pascal-5<sup>1</sup></td>
        <td>Pascal-5<sup>2</sup></td>
        <td>Pascal-5<sup>3</sup></td>
        <td>Mean</td>
    </tr>
    <tr>
        <td rowspan="3">1-shot</td>
        <td>VGG16</td>
        <td>RPMMs</td>
        <td>47.14</td>
        <td>65.82</td>
        <td>50.57</td>
        <td>48.54</td>
        <td>53.02</td>
    </tr>
    <tr>
        <td rowspan="2">Resnet50</td>
        <td>PMMs</td>
        <td>51.98</td>
        <td>67.54</td>
        <td>51.54</td>
        <td>49.81</td>
        <td>55.22</td>
    </tr>
    <tr>
        <td>RPMMs</td>
        <td>55.15</td>
        <td>66.91</td>
        <td>52.61</td>
        <td>50.68</td>
        <td>56.34</td>
    </tr>
    <tr>
        <td rowspan="3">5-shot</td>
        <td>VGG16</td>
        <td>RPMMs</td>
        <td>50.00</td>
        <td>66.46</td>
        <td>51.94</td>
        <td>47.64</td>
        <td>54.01</td>
    </tr>
    <tr>
        <td rowspan="2">Resnet50</td>
        <td>PMMs</td>
        <td>55.03</td>
        <td>68.22</td>
        <td>52.89</td>
        <td>51.11</td>
        <td>56.81</td>
    </tr>
    <tr>
        <td>RPMMs</td>
        <td>56.28</td>
        <td>67.34</td>
        <td>54.52</td>
        <td>51.00</td>
        <td>57.30</td>
    </tr>
</table>

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

## References
Some of our Code is based on the following code:
EMANet:[https://github.com/XiaLiPKU/EMANet](https://github.com/XiaLiPKU/EMANet)
CANet:[https://github.com/icoz69/CaNet](https://github.com/icoz69/CaNet)
SG-One:[https://github.com/xiaomengyc/SG-One](https://github.com/xiaomengyc/SG-One)

