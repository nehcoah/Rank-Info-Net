import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import StepLR

from pathlib import Path

from models.resnet import resnet50
from dataset import SCUT_FBP5500_Dataset
from utils import train_base, test_base

num_classes = 81


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/", help="Directory to save model")
    parser.add_argument("--load_from", type=str, required=True, help="Directory of state-of-dict")
    parser.add_argument("--param", type=list, default=[1., 1., 1., 1.], help="Hyperparameters of loss")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=90)
    parser.add_argument("--method", type=str, default="cross_validation", help="Training method")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multi gpus")
    args = parser.parse_args()
    return args


def cross_validations(split, args):
    if not os.path.exists(os.path.join(args.checkpoint, str(split))):
        os.makedirs(os.path.join(args.checkpoint, str(split)))

    split_train = 'train_test_files/5_folders_cross_validations_files/cross_validation_' + str(split + 1) + '/train_' + str(split + 1) + '.txt'
    split_test = 'train_test_files/5_folders_cross_validations_files/cross_validation_' + str(split + 1) + '/test_' + str(split + 1) + '.txt'

    model = resnet50(num_classes=num_classes)
    checkpoint = torch.load(args.load_from, map_location='cpu')
    p = {k: v for k, v in checkpoint.items() if (k in checkpoint and 'fc' not in k)}
    model.load_state_dict(p, strict=False)

    model = model.to(args.device)

    if args.multi_gpu:
        model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = nn.SmoothL1Loss().to(args.device)

    train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.RandomCrop((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    train_dataset = SCUT_FBP5500_Dataset(args.data_dir, split_train, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor()])
    test_dataset = SCUT_FBP5500_Dataset(args.data_dir, split_test, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.3, last_epoch=-1)

    best = [1000., 1000., 0.]

    for i in range(args.epoch):
        train_base(train_loader, model, criterion, optimizer, num_classes, i, args.device, args.param)
        mae, rmse, corr = test_base(test_loader, model, num_classes, args.device)
        model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
        if mae < best[0]:
            print(f"=> [epoch {i:03d}] best test mae was improved from {best[0]:.4f} to {mae:.4f}")
            torch.save(model_state_dict, str(Path(args.checkpoint).joinpath(str(split)).joinpath("best_mae.pth")))
            best = [mae, rmse, corr]
        else:
            print(f"=> [epoch {i:03d}] best test mae was not improved from {best[0]:.4f} ({mae:.4f})")

        scheduler.step()

    return best


def split_64(args):
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    split_train = 'train_test_files/split_of_60%training and 40%testing/train.txt'
    split_test = 'train_test_files/split_of_60%training and 40%testing/test.txt'

    model = resnet50(num_classes=num_classes)
    checkpoint = torch.load(args.load_from, map_location='cpu')
    p = {k: v for k, v in checkpoint.items() if (k in checkpoint and 'fc' not in k)}
    model.load_state_dict(p, strict=False)

    model = model.to(args.device)

    if args.multi_gpu:
        model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = nn.SmoothL1Loss().to(args.device)

    train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.RandomCrop((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    train_dataset = SCUT_FBP5500_Dataset(args.data_dir, split_train, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor()])
    test_dataset = SCUT_FBP5500_Dataset(args.data_dir, split_test, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=-1)

    best = [1000., 1000., 0.]

    for i in range(args.epoch):
        train_base(train_loader, model, criterion, optimizer, num_classes, i, args.device, args.param)
        mae, rmse, corr = test_base(test_loader, model, num_classes, args.device)
        model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
        if mae < best[0]:
            print(f"=> [epoch {i:03d}] best test mae was improved from {best[0]:.4f} to {mae:.4f}")
            torch.save(model_state_dict, str(Path(args.checkpoint).joinpath("best_mae.pth")))
            best = [mae, rmse, corr]
        else:
            print(f"=> [epoch {i:03d}] best test mae was not improved from {best[0]:.4f} ({mae:.4f})")

        scheduler.step()

    return best


def main():
    args = get_args()
    print(args)

    assert args.method in ['cross_validation', '6-4'], "Unsupported training method."
    assert len(args.param) == 4

    if args.method == '6-4':
        print("Method: " + args.method + " with params " + str(args.param))
        mae, rmse, pc = split_64(args)
        print(f"Mae: {mae:.4f}. Rmse: {rmse:.4f}. PC: {pc:.4f}.")

    else:
        print("Method: " + args.method + " with params " + str(args.param))
        for i in range(5):
            mae, rmse, pc = cross_validations(i, args)
            print(f"Split: {i:02d}. Mae: {mae:.4f}. Rmse: {rmse:.4f}. PC: {pc:.4f}.")


if __name__ == '__main__':
    main()
