import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from models.yolov5 import Model
from losses.loss import ComputeLoss

from datas.dataloaders import create_dataloader, DetectionDataSet
from datas.data_sampler import EnlargedSampler
from datas.prefetch_dataloader import CUDAPrefetcher, CPUPrefetcher

from utils.torch_utils import *
from utils.general import *
from utils.metrics import *


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.nc = opt['nc']
        self.maps = np.zeros(self.nc)

        self.grid_size = 32
        self.img_size = self.opt['img_size']

        self.model = None
        self.compute_loss = None
        self.stopper, self.stop = None, False
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.data_pipline = None
        self.best_fitness = 0.0
        self.total_epochs = 0
        self.start_epoch = 1
        self.prefetcher = None
        self.train_sampler = None
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

    def train_init_setting(self):

        self.current_iter = 0
        self.total_iters = self.opt['max_iters']

        self.total_epochs = self.opt['epochs']

        self.model = Model(self.opt['model_cfg']).to(self.device)
        self.model = self.model_to_device(self.model)

        self.grid_size = max(int(self.model.stride.max()), 32)
        self.img_size = check_img_size(self.opt['img_size'], self.grid_size, floor=self.grid_size * 2)

        self.compute_loss = ComputeLoss(self.model)
        self.stopper = EarlyStopping(patience=self.opt['train']['patience'])
        self.optimizer = create_optimizer(self.model,
                                          name=self.opt['optimizer']['name'],
                                          lr=self.opt['optimizer']['lr'],
                                          momentum=self.opt['optimizer']['momentum'],
                                          decay=self.opt['optimizer']['decay'])

        self.train_dataset = DetectionDataSet(path=self.opt['dataset']['train']['path'],
                                              img_size=self.opt['dataset']['train']['img_size'],
                                              hyp=self.opt['dataset']['train']['hyp'],
                                              rect=self.opt['dataset']['train']['rect'],
                                              image_weights=self.opt['dataset']['train']['image_weights']
                                              )
        self.train_sampler = EnlargedSampler(self.train_dataset, self.opt['world_size'], self.opt['rank'], 1)
        self.train_loader = create_dataloader(self.train_dataset,
                                              sampler=self.train_sampler,
                                              dataset_opt=self.opt['dataset']['train'],
                                              seed=self.opt['dataset']['train']['seed'])

        self.valid_dataset = DetectionDataSet(path=self.opt['dataset']['val']['path'],
                                              img_size=self.opt['dataset']['val']['img_size'],
                                              hyp=self.opt['dataset']['val']['hyp'],
                                              rect=self.opt['dataset']['val']['rect'],
                                              image_weights=self.opt['dataset']['val']['image_weights']
                                              )
        self.valid_loader = create_dataloader(self.valid_dataset,
                                              sampler=None,
                                              dataset_opt=self.opt['dataset']['val'],
                                              seed=self.opt['dataset']['val']['seed'])


        prefetcher_mode = self.opt['datasets']['train'].get('prefetch_mode')
        self.prefetcher = CUDAPrefetcher(self.train_loader, self.opt['num_gpu']) \
            if prefetcher_mode == 'cuda' \
            else CPUPrefetcher(self.train_loader)


    def _run_batch(self, train_data):
        self.optimizer.zero_grad()
        imgs, targets = train_data[0], train_data[1]
        imgs, targets = imgs.to(self.device, non_blocking=True).float() / 255, targets.to(self.device)
        if self.opt['multi_scale']:
            size = random.randrange(
                int(self.img_size * 0.5), int(self.img_size * 1.5)
                                          + self.grid_size) // self.grid_size * self.grid_size

            scale_factor = size / max(imgs.shape[2:])
            if scale_factor != 1:
                new_shape = [math.ceil(x * scale_factor / self.grid_size) for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)

        pred = self.model(imgs)
        loss, loss_items = self.compute_loss(pred, targets)

        loss.backward()
        return loss_items

    def train(self):
        self.model.train()
        if self.opt['image_weights']:
            class_w = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.nc  # class weights
            image_w = labels_to_image_weights(self.train_dataset.labels, nc=self.nc, class_weights=class_w)  # image weights
            self.train_dataset.indices = random.choices(range(self.train_dataset.n), weights=image_w, k=self.train_dataset.n)

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.train_sampler.set_epoch(epoch)
            self.prefetcher.reset()
            train_data = self.prefetcher.next()

            while train_data is not None:
                self.current_iter += 1
                loss_items = self._run_batch(train_data)
                self.optimizer.step()

                if self.current_iter % self.opt['logger']['print_freq'] == 0:
                    print(f"epoch: {epoch}, current iters: {self.current_iter} ===> loss: {loss_items}, "
                          f"lr: {self.opt['optimizer']['lr']}")

                if self.current_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
                    print(f"Saving models and training state.")
                    self.model.save(epoch, self.current_iter)

                if self.current_iter % self.opt['val']['val_freq'] == 0:
                    print(f"epoch: {epoch}, current iters: {self.current_iter}, Validation =====>:.")
                    results, maps = self.valid()

                    # Update best mAP
                    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    self.stop = self.stopper(epoch=epoch, fitness=fi)  # early stop check
                    if fi > self.best_fitness:
                        self.best_fitness = fi

                train_data = self.prefetcher.next()

            if self.stop:
                break

    def process_batch(self, detections, labels, iouv):
        """
        :param detections: array[N, 6], x1, y1, x2, y2, conf, class
        :param labels: array[M, 5], class, x1, y1, x2, y2
        :param iouv:
        :return: correct prediction matrix array[N, 10], for 10 IoU levels
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where(iou > iouv[i] & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy() # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def valid(self):
        self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        niou = iouv.numel()

        seen = 0
        val_opt = self.opt['val']
        confusion_matrix = ConfusionMatrix(nc=self.nc)
        names = self.model.names if hasattr(self.model, 'name') else self.model.module.names
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        class_map = list(range(1, self.nc + 10))
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        loss = torch.zeros(3, device=self.device)
        jdict, stats, ap, ap_class = [], [], [], []
        pbar = tqdm(self.valid_loader, desc=s, bar_format=TQDM_BAR_FORMAT)

        for batch_i, (im, targets) in enumerate(pbar):
            with torch.no_grad():
                im, targets = im.to(self.device), targets.to(self.device)

            im /= 255
            nb, _, height, width = im.shape  # batch size, channels, height, width
            preds = self.model(im) if val_opt['compute_loss']else (
                self.model(im, augment=val_opt['augment']), None)

            if val_opt['compute_loss']:
                loss += self.compute_loss(preds, targets)[1]

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if val_opt['save_hybrid'] else []

            preds = non_max_suppression(preds,
                                        val_opt['conf_thres'],
                                        val_opt['iou_thres'],
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=val_opt['single_cls'],
                                        max_det=val_opt['max_det'])

            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=self.device)
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                        if val_opt['plots']:
                            confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    continue

                # Predictions
                if val_opt['single_cls']:
                    pred[:, 5] = 0
                predn = pred.clone()
                # scale_boxes(im[si].shape[1:], predn[:, :4])

                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = self.process_batch(predn, labelsn, iouv)
                    if val_opt['plots']:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats,
                                                          plot=val_opt['plots'],
                                                          save_dir=val_opt['save_dir'],
                                                          names=names)
            ap50, ap = ap[:, 0], ap.mean(1) #AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.nc)

        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        if nt.sum() == 0:
            print(f'WARNING ⚠️ no labels found in val set, can not compute metrics without labels')

        maps = np.zeros(self.nc) + map

        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        self.model.train()
        return (mp, mr, map50, map, *(loss.cpu() / len(self.valid_loader)).tolist()), maps

    def get_model_attr(self):
        nl = de_parallel(self.model).model[-1].nl
        self.opt['hyp']['box'] *= 3 / nl
        self.opt['hyp']['cls'] *= self.nc / 80 * 3 / nl
        self.opt['hyp']['obj'] *= (self.opt['img_size'] / 640) ** 2 * 3 / nl
        self.model.nc = self.nc
        self.model.hyp = self.opt['hyp']
        self.model.names = self.opt['names']
        self.model.class_weights = labels_to_class_weights(
                                            self.train_dataset.labels,
                                            self.nc).to(self.device) * self.nc  # attach class weights

    def model_to_device(self, model):
        model = model.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            model = DistributedDataParallel(
                model, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            model = DataParallel(model)
        return model


    def save(self):
        pass

    def load_model_dict(self):
        pass

    def load_state_dict(self):
        pass
