import os
import math
import pandas as pd
from typing import Optional, Any, Tuple, List, Dict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def train_base(train_loader, model, criterion, optimizer, num_classes, epoch, device, param) -> None:
    model.train()
    scores = torch.arange(0, num_classes).to(device)

    lambda1, lambda2, lambda3, lambda4 = param
    th = 0.95
    if epoch > 15:
        th = 0.98

    for index, (image, label, std) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        std = std.to(device)

        optimizer.zero_grad()
        logits, rank, cls = model(image)

        pred = F.softmax(logits, dim=1)
        pred = (pred * scores).sum(axis=-1)

        cls_loss = criterion(pred, label)
        d_loss = dis_loss(F.softmax(cls, dim=1), std)
        rank_loss1 = calculate_pair_loss1(rank, label, F.softmax(logits, dim=1), num_classes, device, th)
        rank_loss2 = calculate_pair_loss2(rank, label, num_classes, device)
        loss = lambda1 * rank_loss1 + lambda2 * rank_loss2 + lambda3 * d_loss + lambda4 * cls_loss
        loss.backward()
        optimizer.step()

        print('[%d, %5d] cls_loss: %.3f, dis_loss: %.3f, rank1: %.3f, rank2: %.3f' % (
            epoch, index + 1, cls_loss.item(), d_loss.item(), rank_loss1.item(), rank_loss2.item()))


def test_base(test_loader, model, num_classes, device):
    model.eval()
    scores = np.arange(0, num_classes)
    mae = 0.
    rmse = 0.
    preds = []
    labels = []

    with torch.no_grad():
        for i, (image, label, std) in enumerate(test_loader):
            image = image.to(device)

            output, _, _ = model(image)
            out = F.softmax(output, dim=1).cpu()
            pred = (out * scores).sum(axis=-1).item()

            preds.append(pred)
            labels.append(label.item())
            mae += abs(pred - label.item()) / 20
            rmse += (pred - label.item()) * (pred - label.item()) / 400

    mae = mae / len(test_loader)
    rmse = math.sqrt(rmse / len(test_loader))
    preds_for_pd = pd.Series(preds)
    labels_for_pd = pd.Series(labels)
    corr = preds_for_pd.corr(labels_for_pd, method='pearson')

    print(f"mae: {mae:.4f}, rmse: {rmse:.4f}, pc: {corr:.4f}")
    return mae, rmse, corr


def get_pairs(scores: List[int]) -> List[Tuple]:
    pairs = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
            elif scores[i] < scores[j]:
                pairs.append((j, i))
    return pairs


def split_pairs(order_pairs: List[Tuple]) -> Tuple[List[int], List[int]]:
    relevant_doc = []
    irrelevant_doc = []
    query_num = len(order_pairs)
    for i in range(query_num):
        d1, d2 = order_pairs[i]
        relevant_doc.append(d1)
        irrelevant_doc.append(d2)

    return relevant_doc, irrelevant_doc


def calculate_pair_loss1(output, labels, pred, num_classes, device, threshold=0.9):
    loss = torch.zeros((1,)).to(device)
    for feature, label, p in zip(output, labels, pred):
        label = int(torch.round(label).item())
        info, cur, i = [(0, label)], p[label].item(), 0
        while cur < threshold:
            i += 1
            if label - i >= 0:
                info.append((i, label - i))
                cur += p[label - i].item()
            if label + i < num_classes:
                info.append((i, label + i))
                cur += p[label + i].item()
        i += 1
        while label - i >= 0 or label + i < num_classes:
            if label - i >= 0:
                info.append((num_classes, label - i))
            if label + i < num_classes:
                info.append((num_classes, label + i))
            i += 1

        l_index, s_index = [], []
        for tmp1 in info:
            for tmp2 in info:
                if tmp1[0] < tmp2[0]:
                    l_index.append(tmp1[1])
                    s_index.append(tmp2[1])
        if len(l_index) == 0:
            continue

        fea1 = feature[l_index]
        fea2 = feature[s_index]
        y = np.ones(fea1.shape[0])
        y = torch.Tensor(y).to(device)
        y_pred = torch.sigmoid(fea1 - fea2)
        loss += F.binary_cross_entropy(y_pred, y)

    return loss / len(labels)


def calculate_pair_loss2(output, labels, num_classes, device):
    pair = get_pairs(labels)
    relevant_index, irrelevant_index = split_pairs(pair)

    pred = F.softmax(output, dim=1)
    ages = torch.arange(0, num_classes).to(device)
    pred = (pred * ages).sum(-1)

    relevant_feature = pred[relevant_index]
    irrelevant_feature = pred[irrelevant_index]

    y = np.ones((relevant_feature.shape[0]))
    y = torch.Tensor(y).to(device)

    y_pred = torch.sigmoid(relevant_feature - irrelevant_feature)

    pair_loss = F.binary_cross_entropy(y_pred, y)
    return pair_loss


def dis_loss(pred, dis):
    loss = 0.
    for p, d in zip(pred, dis):
        loss += torch.dist(p, d, p=2)
    return loss / dis.shape[0]


def exp_loss(output, labels):
    return torch.exp(torch.abs((output - labels) / 7)).mean() - 1
