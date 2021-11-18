import argparse
import glob
import importlib as imp
import os
import os.path as osp
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.append("../")
from metrics.measure import IQA


def parse_argumnets():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res_dir", type=str, default=None, help="directory of test images"
    )
    parser.add_argument(
        "--ref_dir", type=str, default=None, help="directory of reference images"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="directory of saved results"
    )
    parser.add_argument("--metrics", type=list, default=["psnr", "ssim", "lpips", "niqe", "piqe", "brisque"])

    args = parser.parse_args()

    return args


def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = (
            np.matmul(
                img,
                [
                    [24.966, 112.0, -18.214],
                    [128.553, -74.203, -93.786],
                    [65.481, -37.797, 112.0],
                ],
            )
            / 255.0
            + [16, 128, 128]
        )
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def main():
    args = parse_argumnets()
    if args.save_dir is None:
        args.save_dir = args.res_dir
    if args.res_dir is None:
        raise TypeError("res dir can not be None")
        if not osp.exists(args.res_dir):
            raise ValueError("res dir dose not exist")

    res_paths = sorted(glob.glob(osp.join(args.res_dir, "*.png")))
    print(f"{len(res_paths)} images to be tested")
    if args.ref_dir is not None:
        ref_paths = sorted(glob.glob(osp.join(args.ref_dir, "*.png")))

        if not len(res_paths) == len(ref_paths):
            raise ValueError(
                f"Number of res images {len(res_paths)} must be equal\
                to Number of ref images {len(ref_paths)}"
            )

    score_file_name = "_".join(osp.abspath(args.res_dir).split("/"))
    score_file_name = osp.join(args.save_dir, f"{score_file_name}.txt")
    score_file = open(score_file_name, "w")

    measure = IQA(metrics=args.metrics, cuda=False)
    test_results_rgb = defaultdict(list)
    test_results_y = defaultdict(list)
    for indx, res_path in enumerate(res_paths):
        res_img = cv2.imread(res_path)

        message = f"image {res_path}\t"
        if args.ref_dir is not None:
            ref_img = cv2.imread(ref_paths[indx])
        else:
            ref_img = None

        message += "Original Scores\t"
        scores = measure(res=res_img, ref=ref_img, metrics=args.metrics)
        for k, v in scores.items():
            test_results_rgb[k].append(v)
            message += "{}: {:.6f}; ".format(k, v)

        if res_img.ndim == 3:
            res_img_y = bgr2ycbcr(res_img, only_y=True)

            if ref_img is not None:
                ref_img_y = bgr2ycbcr(ref_img, only_y=True)
            else:
                ref_img_y = None

            message += "Y Scores\t"
            scores = measure(res=res_img_y, ref=ref_img_y, metrics=args.metrics)
            for k, v in scores.items():
                test_results_y[k].append(v)
                message += "{}: {:.6f}; ".format(k, v)

        print(message)
        score_file.write(message + "\n")

    message = "-" * 10 + "Average Results" + "-" * 10 + "\n"
    message += "Origianl Scores\t"
    for k, v in test_results_rgb.items():
        ave = sum(v) / len(v)
        message += "{}: {:.6f}; ".format(k, ave)

    if len(test_results_y) > 0:
        message += "Y Scores\t"
        for k, v in test_results_y.items():
            ave = sum(v) / len(v)
            message += "{}: {:.6f}; ".format(k, ave)

    print(message)
    score_file.write(message)
    score_file.close()


if __name__ == "__main__":
    main()
