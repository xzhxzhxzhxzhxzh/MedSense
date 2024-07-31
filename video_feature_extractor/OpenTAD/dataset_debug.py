import os
import sys
import argparse
import torch
from mmengine.config import Config
from opentad.datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    args = parser.parse_args()
    return args

args = parse_args()
# load config
cfg = Config.fromfile(args.config)
test_dataset = build_dataset(cfg.dataset.test)

result = test_dataset.__getitem__(1)
print("Done!")