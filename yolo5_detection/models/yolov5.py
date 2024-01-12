from copy import deepcopy
import torch
import torch.nn as nn
import math
from net_parser import parse_model
from utils.torch_utils import scale_img



class Model(nn.Module):
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super(Model, self).__init__()

        if isinstance(cfg, dict):
            self.net_cfg = cfg
        else:
            import yaml
            with open(cfg) as f:
                self.net_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        ch = self.net_cfg['chs'] = self.net_cfg.get('ch', ch)

        if nc and nc != self.net_cfg['nc']:
            print(f"Overriding model.yaml nc = {self.net_cfg['nc']} with nc={nc}")
            self.net_cfg['nc'] = nc
        if anchors:
            print(f"Overriding model.yaml anchors with anchors={anchors}")

        self.model, self.save = parse_model(deepcopy(self.net_cfg), ch=[ch])
        self.names = [str(i) for i in range(self.net_cfg['nc'])]
        self.inplace = self.net_cfg.get('inplace', True)

        detector = self.model[-1]
        s = 256
        detector.inplace = self.inplace
        forward = lambda x: self.forward(x)
        detector.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
        detector.anchor /= detector.stride.view(-1, 1, 1)
        self.stride = detector.stride

    def forward(self, x, augment=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.filp(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, dim=1), None

    def _forward_once(self, x):
        y, dt = [], []
        for i, module in enumerate(self.model):
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
            # print(f"wanchao   {i}: {module}, x.shape: {x.shape}")
            x = module(x)
            y.append(x if module.i in self.save else None)

        return x

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] -y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4 ** x for x in range(nl))
        e = 1
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))
        y[0] = y[0][:, :-i]
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][:, i:]
        return y


if __name__ == "__main__":
    yolo = Model(cfg="/home/xm/disk1/wanchao/CV_learning/detection/net_config/yolov5s.yaml")
    x = torch.randn(1, 3, 640, 640)
    y = yolo(x)
    print(f"wanchao: \n "
          f"{y[0].shape}\n"
          f"{y[1].shape}\n"
          f"{y[2].shape}")
