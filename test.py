import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.resnet import resnet50
from dataset import SCUT_FBP5500_Dataset
from utils import test_base


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--load_from", type=str, required=True, help="Directory of state-of-dict")
    parser.add_argument("--method", type=str, default="cross_validation", help="Training method")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    num_classes = 81

    assert args.method in ['cross_validation', '6-4'], "Unsupported testing method."

    print("Testing:")
    model = resnet50(num_classes=num_classes)

    if args.method == '6-4':
        checkpoint_path = args.load_from + '/best_mae.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.to(args.device)

        split_test = 'train_test_files/split_of_60%training and 40%testing/test.txt'
        test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop((224, 224)),
                                             transforms.ToTensor()])
        test_dataset = SCUT_FBP5500_Dataset(args.data_dir, split_test, test_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        test_base(test_loader, model, num_classes, args.device)

    else:
        for split in range(5):
            checkpoint_path = args.load_from + '/' + str(split) + '/best_mae.pth'

            split_test = 'train_test_files/5_folders_cross_validations_files/cross_validation_' + str(split + 1) + '/test_' + str(split + 1) + '.txt'

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            model = model.to(args.device)

            test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                                 transforms.CenterCrop((224, 224)),
                                                 transforms.ToTensor()])
            test_dataset = SCUT_FBP5500_Dataset(args.data_dir, split_test, test_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

            print(f'Split {split:01d}:')
            test_base(test_loader, model, num_classes, args.device)


if __name__ == '__main__':
    main()
