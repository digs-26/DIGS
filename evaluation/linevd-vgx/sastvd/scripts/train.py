import os
import sys
import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune
import pandas as pd
import pytorch_lightning as pl
import sastvd.linevd as lvd
import pdb
import argparse
import time
import random
import numpy as np
import torch
import dgl
import config
#from ray.tune import Analysis

os.environ["SLURM_JOB_NAME"] = "bash"

linevd_config = {
    "hfeat": 512,
    "embtype": "codebert",
    "stmtweight": 1,
    "hdropout": 0.3,
    "gatdropout": 0.2,
    "modeltype": "gat2layer",
    "gnntype": "gat",
    "loss": "ce",
    "scea": 0.5,
    "gtype": "pdg+raw",
    "batch_size": 1024,
    "multitask": "linemethod",
    "splits": "default",
    "lr": 1e-4
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bigvul_vgx_under0.8_random", type=str)
    parser.add_argument("--trainset", default="bigvul", type=str)
    parser.add_argument("--partset", default="vgx", type=str)
    parser.add_argument("--testset", default="reveal", type=str)
    parser.add_argument("--seed", default=123456, type=int)
    parser.add_argument("--under", default=0.8, type=float)
    parser.add_argument("--selection", default="random", type=str)
    args = parser.parse_args()

    set_seed(args)
    config.set_seed(args.seed)
    config.set_trainset(args.trainset)
    config.set_partset(args.partset)
    config.set_testset(args.testset)
    config.set_under(args.under)
    config.set_selection(args.selection)

    indices_path = os.path.join("./storage/indices", args.trainset + "_line_" + args.partset + "_line_123456_under" + str(args.under) + "_over_" + args.selection, "coreset_index.txt")
    config.set_indices_path(indices_path)


    start = time.time()
    print("Start training: ", start)
    samplesz = -1
    sp = os.path.join("./storage/checkpoint", args.dataset, str(args.seed))
    os.makedirs(sp, exist_ok=True)
    lvdrun.train_linevd(config=linevd_config, max_epochs=130, samplesz=samplesz, savepath=sp)
    end = time.time()
    print("End training: ", end)
    print("Time taken: ", end - start)



if __name__ == "__main__":
    main()
