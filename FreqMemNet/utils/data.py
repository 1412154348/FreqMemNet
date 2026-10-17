import os.path as osp
import random

import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps


class IRSTDDataset(Data.Dataset):
    def __init__(self, args, mode='train'):
        self.dataset_name = args.dataset_name

        dataset_dir = osp.join(args.data_root, args.dataset_name)
        txtfile = 'trainval.txt' if mode == 'train' else 'test.txt'
        self.suffix = '.jpg' if self.dataset_name == 'SIRST-UAVB' else '.png'

        self.list_dir = osp.join(dataset_dir, txtfile)
        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')

        with open(self.list_dir, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size

        if self.dataset_name == 'IRSTD-1k':
            mean = [.343, .343, .343]
            std = [.230, .230, .230]
        elif self.dataset_name == 'NUDT-SIRST':
            mean = [.424, .424, .424]
            std = [.218, .218, .218]
        elif self.dataset_name == 'NUAA-SIRST':
            mean = [.449, .448, .449]
            std = [.224, .224, .224]
        else:
            mean = [.343, .343, .343]
            std = [.230, .230, .230]

        self.mask_transform = transforms.ToTensor()
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + self.suffix)
        label_path = osp.join(self.label_dir, name + self.suffix)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        temp = self.mask_transform(mask)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._basic_transform(img, mask)

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return img, mask, temp

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        long_size = random.randint(
            int(self.base_size * 0.5), int(self.base_size * 2.0)
        )
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask

    def _basic_transform(self, img, mask):
        img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        return img, mask


__all__ = ['IRSTDDataset']