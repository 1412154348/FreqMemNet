from argparse import ArgumentParser
import bisect
import os
import os.path as osp
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import AdamW, Adagrad, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.freqmemnet import FreqMemNet, ResidualBlock
from utils.data import IRSTDDataset
from utils.losses import AverageMeter, IoULoss
from utils.metric import PD_FA, ROCMetric, SamplewiseSigmoidMetric, SigmoidMetric


def parse_args():
    parser = ArgumentParser(description='Train or evaluate RDANet')

    parser.add_argument(
        '--dataset-name',
        type=str,
        default='NUAA-SIRST',
        choices=['IRSTD-1k', 'NUDT-SIRST', 'NUAA-SIRST'],
        help='dataset used for training or evaluation',
    )
    parser.add_argument('--data-root', type=str, default='./datasets',
                        help='root directory containing the dataset folders')
    parser.add_argument('--save-root', type=str, default='./runs',
                        help='directory used to save training outputs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--eta-min', type=float, default=1e-5)

    parser.add_argument('--base-size', type=int, default=512)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--multi-gpus', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='training checkpoint used when resuming')

    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'Adam', 'Adagrad'])

    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--block-factor', type=int, default=1)

    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--weight-path', type=str, default=None,
                        help='checkpoint used in test mode')

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.mode = args.mode

        valset = IRSTDDataset(args, mode='val')

        if self.mode == 'train':
            trainset = IRSTDDataset(args, mode='train')
            self.train_loader = Data.DataLoader(
                trainset, args.batch_size, num_workers=0, shuffle=True
            )
            self.val_loader = Data.DataLoader(
                valset, 4, num_workers=0, shuffle=False
            )
        else:
            self.val_loader = Data.DataLoader(valset, 1, num_workers=4, shuffle=False)

        device = torch.device('cuda')
        self.device = device

        if args.dataset_name == 'IRSTD-1k':
            mem_slots = [256, 128, 64, 32]
            patch = [8, 8, 8, 4]
            topk = [8, 8, 8, 4]
            drop_out = [0, 0, 0, 0.0]
            use_overlap = [False, False, False, False]
        elif args.dataset_name == 'NUDT-SIRST':
            mem_slots = [256, 128, 64, 32]
            patch = [16, 8, 8, 4]
            topk = [16, 8, 8, 4]
            drop_out = [0, 0, 0, 0.0]
            use_overlap = [False, False, False, False]
        elif args.dataset_name == 'NUAA-SIRST':
            mem_slots = [256, 128, 64, 32]
            patch = [16, 8, 8, 4]
            topk = [16, 8, 8, 4]
            drop_out = [0, 0, 0, 0.0]
            use_overlap = [False, False, False, False]
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")

        model = FreqMemNet(
            3,
            args.base_size,
            args.layers,
            args.factor,
            args.block_factor,
            block=ResidualBlock,
            mem_slots=mem_slots,
            patch=patch,
            topk=topk,
            drop_out=drop_out,
            use_overlap=use_overlap,
        )
        model.to(device)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use ' + str(torch.cuda.device_count()) + ' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        print('use ' + str(torch.cuda.device_count()) + ' gpus')

        self.model = model

        if args.optimizer == 'Adagrad':
            self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.06)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                   weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=600, eta_min=args.eta_min
        )

        self.loss_fun = IoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = SigmoidMetric()
        self.nIoU = SamplewiseSigmoidMetric(1)
        self.ROC = ROCMetric(1, 10)
        self.best_iou = 0

        self.BIN_LABELS = ['[1,15]', '[16,31]', '[32,63]', '[64,127]', '[128,255]', '[256,∞)']
        self.AREA_EDGES = [15, 31, 63, 127, 255]

        self.bin_metrics = [SigmoidMetric() for _ in range(6)]
        self.bin_counts = [0 for _ in range(6)]

        if args.mode == 'train':
            if args.checkpoint_path is not None:
                checkpoint_path = osp.abspath(args.checkpoint_path)
                check_folder = osp.dirname(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                run_name = args.dataset_name + '-%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
                self.save_folder = osp.join(args.save_root, run_name)
                os.makedirs(self.save_folder, exist_ok=False)
                dict_args = vars(args)
                args_key = list(dict_args.keys())
                args_value = list(dict_args.values())
                with open(self.save_folder + '/train_log.txt', 'w') as f:
                    now = datetime.now()
                    f.write("time:--")
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    f.write(dt_string)
                    f.write('\n')
                    for i in range(len(args_key)):
                        f.write(args_key[i])
                        f.write(':--')
                        f.write(str(args_value[i]))
                        f.write('\n')
            self.writer = SummaryWriter(log_dir=osp.join(self.save_folder, 'tensorboard'))
        if args.mode == 'test':
            if args.weight_path is None:
                raise ValueError('--weight-path is required in test mode')
            try:
                state_dict = torch.load(
                    args.weight_path, map_location=self.device, weights_only=True
                )
            except TypeError:
                state_dict = torch.load(args.weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()

        for i, (data, mask, temp) in enumerate(tbar):
            data = data.to(self.device)
            mask = mask.to(self.device)

            pred = self.model(data)
            loss = self.loss_fun(pred, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), data.size(0))
            tbar.set_description(
                'Epoch %d, lr %.6f, loss %.4f' % (epoch, self.optimizer.param_groups[0]['lr'], losses.avg))
        self.writer.add_scalar('Loss/loss', losses.avg, epoch)

        if epoch < 600:
            self.scheduler.step()

    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.nIoU.reset()

        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader)
        count = 0
        with torch.no_grad():
            for i, (data, mask, temp) in enumerate(tbar):
                data = data.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(data)
                count = count + data.size(0)

                pred = pred.cpu()
                mask = mask.cpu()
                self.mIoU.update(pred, mask)
                self.nIoU.update(pred, mask)

                if self.mode == 'test':
                    from scipy.ndimage import label

                    lbl, num = label(temp)
                    if num == 0:
                        area = 0
                    else:
                        cnt = np.bincount(lbl.ravel())[1:]
                        area = int(cnt.max())
                    if area >= 1:
                        idx = bisect.bisect_right(self.AREA_EDGES, area)
                        self.bin_metrics[idx].update(pred, mask)
                        self.bin_counts[idx] += 1

                    self.PD_FA.update(pred, mask)
                    self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()
                _, n_IoU = self.nIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f, nIoU %.4f' % (epoch, mean_IoU, n_IoU))

            _, mean_IoU = self.mIoU.get()
            _, n_IoU = self.nIoU.get()

            if self.mode == 'train':
                iou = mean_IoU
                self.writer.add_scalar('IoU/mean', mean_IoU, epoch)
                self.writer.add_scalar('IoU/nIoU', n_IoU, epoch)
                if iou > self.best_iou:
                    self.best_iou = iou
                    all_states = {"net": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                                  'scheduler': self.scheduler.state_dict(), "epoch": epoch, "iou": self.best_iou}
                    torch.save(self.model.state_dict(), self.save_folder + '/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - nIoU {:.4f}\n'.
                                format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                       epoch, mean_IoU, n_IoU))
                    torch.save(all_states, self.save_folder + '/best_checkpoint.pkl')
                all_states = {"net": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                              'scheduler': self.scheduler.state_dict(), "epoch": epoch, "iou": self.best_iou}
                torch.save(all_states, self.save_folder + '/checkpoint.pkl')

            elif self.mode == 'test':
                FA, PD = self.PD_FA.get(count)
                true_positive_rate, false_positive_rate, _, _ = self.ROC.get()
                print('mIoU: ' + str(mean_IoU) + '\n')
                print('nIoU:', n_IoU)
                print('Pd: ' + str(PD[0]) + '\n')
                print('Fa: ' + str(FA[0] * 1000000) + '\n')
                print('tp:', true_positive_rate)
                print('fp:', false_positive_rate)
                print('\n# Bin-wise IoU:')
                for k, metric in enumerate(self.bin_metrics):
                    _, mIoU_k = metric.get()
                    print(f'{self.BIN_LABELS[k]:>8} | num={self.bin_counts[k]:4d} | mIoU={float(mIoU_k):.4f}')


if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)

    if trainer.mode == 'train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
        trainer.writer.close()
    else:
        trainer.test(1)


