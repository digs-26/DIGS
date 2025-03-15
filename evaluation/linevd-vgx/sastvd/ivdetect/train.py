"""Implementation of IVDetect."""


import pickle as pkl
from importlib import reload

import dgl
import sastvd as svd
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.ivdetect.evaluate as ivde
import sastvd.ivdetect.gnnexplainer as ge
import sastvd.ivdetect.helpers as ivd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import pdb
import sastvd.helpers.datasets as svdds
import argparse
import random
import numpy as np
import time
import os
import config
import dgl
import pandas as pd
from tqdm import tqdm


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


def top_k_accuracy(pred_lines, correct_lines, k):
    correct_count = 0
    total_count = 0

    for sampleid, pred in pred_lines.items():
        true = correct_lines[sampleid]
        true = list(true["removed"]) + list(true["depadd"])
        # 获取 Top-k 预测
        top_k_preds = pred[:k]
        # print("True: ", true, "\t\tPred: ", top_k_preds)

        # 检查前 k 个预测中是否有正确答案
        if any(p in true for p in top_k_preds):
            correct_count += 1

        total_count += 1

    top_k_accuracy = correct_count / total_count
    return top_k_accuracy


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

    indices_path = os.path.join("./storage/indices",
                                args.trainset + "_line_" + args.partset + "_line_123456_under" + str(
                                    args.under) + "_over_" + args.selection, "coreset_index.txt")
    config.set_indices_path(indices_path)

    start = time.time()
    print("Start training: ", start)

    # Load data
    reload(ivd)
    df = svdds.bigvul()
    df['id'] = df['id'].astype(str)
    train_ds = ivd.BigVulDatasetIVDetect(partition="train", df=df)
    test_ds = ivd.BigVulDatasetIVDetect(partition="test", df=df)
    dl_args = {"drop_last": False, "shuffle": True}
    train_dl = GraphDataLoader(train_ds, batch_size=16, **dl_args)
    test_dl = GraphDataLoader(test_ds, batch_size=64, **dl_args)

    # %% Create model
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    svd.debug(dev)
    model = ivd.IVDetect(200, 64)
    model.to(dev)

    # %% Optimiser
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    res = []
    # Train loop
    sp = os.path.join("./storage/checkpoint_ivdetect", args.dataset, str(args.seed))
    os.makedirs(sp, exist_ok=True)
    for epoch in range(1):
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}"):
            # Training
            model.train()
            batch = batch.to(dev)
            logits = model(batch, train_ds)
            labels = dgl.max_nodes(batch, "_VULN").long()
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save checkpoint
        checkpoint_path = os.path.join(sp, f"epoch_{epoch}.ckpt")
        torch.save(model.state_dict(), checkpoint_path)


        # %% Statement-level through GNNExplainer
        correct_lines = ivde.get_dep_add_lines_bigvul()
        pred_lines = dict()
        for batch in tqdm(test_dl, desc=f"Epoch {epoch} test"):
            for g in dgl.unbatch(batch):
                idx = g.ndata["_SAMPLE"].max().int().item()
                sampleid = test_ds.idx2id[idx]
                if sampleid not in correct_lines:
                    continue
                if sampleid in pred_lines:
                    continue
                try:
                    lines = ge.gnnexplainer(model, g.to(dev), test_ds)
                    pred_lines[sampleid] = lines
                # lines = ge.gnnexplainer(model, asts, test_ds)
                except Exception as E:
                    print(E)

        data = []
        for sampleid in pred_lines:
            if sampleid in correct_lines:
                data.append({
                    "sampleid": sampleid,
                    "pred_lines": pred_lines[sampleid],
                    "correct_lines": correct_lines[sampleid]
                })
        pred = pd.DataFrame(data)
        pred = pred.sort_values(by="sampleid")
        pred.to_json(os.path.join(sp, f"epoch_{epoch}.jsonl"), orient='records', lines=True)


        # 测试 Top-k 准确率
        k = 10  # 可以根据需要修改 k
        accuracy = top_k_accuracy(pred_lines, correct_lines, k)
        print(f"epoch_{epoch} Top-{k} Accuracy: {accuracy:.4f}")
        res.append({"epoch": epoch, "Top-k Accuracy": accuracy})
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(sp, "results.csv"), index=False)

    end = time.time()
    print("End training: ", end)
    print("Time: ", end - start)


if __name__ == "__main__":
    main()