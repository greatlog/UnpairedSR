import argparse
import sys

import torch
from torchsummaryX import summary

sys.path.append("../../")
import utils.option as option
from models import create_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--opt",
    type=str,
    default="options/setting1/test/test_setting1_x4.yml",
    help="Path to option YMAL file of Predictor.",
)
args = parser.parse_args()
opt = option.parse(args.opt, root_path=".", is_train=True)

opt = option.dict_to_nonedict(opt)
model = create_model(opt)

test_tensor = torch.randn(1, 3, 270, 180).cuda()
for name, net in model.networks.items():
    summary(net.cuda(), x=test_tensor)
    print("Above are results for net {}".format(name))
    input()
