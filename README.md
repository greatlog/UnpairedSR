This is an offical implementation of [PDM-SR](codes/config/PDMSR/). This repo also contains the implementations of many other blind SR methods in [config](codes/config/)

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