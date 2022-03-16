This is an offical implementation of the CVPR2022's paper [Learning the Degradation Distribution for Blind Image Super-Resolution](https://arxiv.org/abs/2203.04962). This repo also contains the implementations of many other blind SR methods in [config](codes/config/), including CinGAN, CycleSR, DSRGAN-SR, etc.

If you find this repo useful for your work, please cite our paper:
```
@inproceedings{PDMSR,
  title={Learning the Degradation Distribution for Blind Image Super-Resolution},
  author={Zhengxiong Luo and Yan Huang and and Shang Li and Liang Wang and Tieniu Tan},
  booktitle={CVPR},
  year={2022}
}
```

The codes are built on the basis of [BasicSR](https://github.com/xinntao/BasicSR).

## Dependences
1. lpips (pip install --user lpips)
2. matlab (to support the evaluation of NIQE). The details about installing a matlab API for python can refer to [here](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

## Datasets
The datasets in NTIRE2017 and NTIRE2018 can be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/). The datasets in NTIRE2020 can be downloaded from the [competition site](https://competitions.codalab.org/competitions/22220).

## Start up
We provide the [checkpoints](https://pan.baidu.com/s/1Ju8szYbK4CmEpYac-wUs4g?pwd=qfe5)(password: qfe5) trained for the track2 of NTIRE2020. To get a quick start:

```bash
cd codes/config/PDM-SR/
python3 inference.py --opt options/test/2020Track2.yml
```