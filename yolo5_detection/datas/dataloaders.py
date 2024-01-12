import glob
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
import cv2
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from augmentations import *
from utils.general import xywhn2xyxy, xyxy2xywhn
from utils.dist_util import *



def get_work_init_fn(num_workers, seed):
    if seed is None:
        return None
    else:
        rank, _ = get_dist_info()
        def worker_init_fn(worker_id):
            # Set the worker seed to num_workers * rank + worker_id + seed
            worker_seed = num_workers * rank + worker_id + seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        return worker_init_fn

def create_dataloader(dataset, dataset_opt, sampler=None, seed=None):
    batch_size = min(dataset_opt['batch_size'], len(dataset))
    if dataset_opt['rect'] and dataset_opt['shuffle']:
        shuffle = False
    else:
        shuffle = True

    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, dataset_opt['num_workers']])
    worker_init_fn = get_work_init_fn(nw, seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle and sampler is None,
                                         drop_last=True, pin_memory=True, worker_init_fn=worker_init_fn, num_workers=nw)
    return loader
    # generator = torch.Generator()
    # generator.manual_seed(10000000 + seed + RANK)


class DetectionDataSet(Dataset):
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    def __init__(self,
                 path,
                 img_size=640,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 stride=32):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        img_names = os.listdir(self.path['images'])
        labels_name = [img.replace('jpg', 'txt') for img in img_names]
        self.img_files = [os.path.join(self.path['images'], img) for img in img_names]
        self.label_files = [os.path.join(self.path['labels'], label) for label in labels_name]
        self.labels = [self.load_label(i) for i in range(len(self.label_files))]
        self.n = len(self.img_files)
        self.indices = range(self.n)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            img, labels = self.load_mosaic(index)
            shapes = None

            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            labels = self.labels[index].copy()
            if labels.shape:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
        if self.augment:
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::, -1] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes


    def load_label(self, index):
        lb_file = self.label_files[index]
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    msg = f'WARNING ⚠️ {lb_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        return lb

    def load_image(self, i):
        img_path = self.img_files[i]
        img = cv2.imread(img_path)
        assert img is not None, f"Image Not Found {img_path}"
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            img = cv2.resize(img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]

    def load_mosaic(self, index):
        labels4 = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        img4 = None
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            labels = self.labels[index].copy()
            if labels.shape:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)

        labels4 = np.concatenate(labels4, 0)
        for x in labels4[:, 1:]:
            np.clip(x, 0, 2 * s, out=x)

        # Augment
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)
        return img4, labels4

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.stack(label, 0)

    @staticmethod
    def collate_fn4(batch):
        im, label = zip(*batch)
        im4, label4 = [], []
        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
        n = len(label) // 4
        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(),
                                    scale_factor=2.0, mode='bilinear', align_corners=False)[0].type(im[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i
        return torch.stack(im4, 0), torch.cat(label4, 0)
