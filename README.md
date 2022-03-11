This is an offical implementation of the CVPR2022's paper [Learning the Degradation Distribution for Blind Image Super-Resolution](https://arxiv.org/abs/2203.04962). This repo also contains the implementations of many other blind SR methods in [config](codes/config/)

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

## Training
```bash
cd codes/config/PDMSR/
python3 train.py --opt options/train/psnr/2017Track1.yml
```

## Testing
```bash
cd codes/config/PDMSR/
python3 test.py --opt options/test/2017Track1.yml
```