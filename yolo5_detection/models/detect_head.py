import torch
import torch.nn as nn
import torch.nn.functional as F


class Detect(nn.Module):
    # yolov5 Detect head for detection models
    stride = None
    dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super(Detect, self).__init__()
        self.nc = nc # number of classes
        self.no = nc + 5 # number of output per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2 # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer("anchor", torch.tensor(anchors).float().view(self.nl, -1, 2)) # shape(nl, na, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch) # output conv
        self.inplace = inplace

    def forward(self, x):
        z = [] # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape # x(bs, 255, 20, 20) to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training: # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i] # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        device = self.anchor_grid[i].device
        d_type = self.anchor_grid[i].dtype
        shape = 1, self.na, ny, nx, 2 # grid shape
        y, x = torch.arange(ny, device=device, dtype=d_type), torch.arange(nx, device=device, dtype=d_type)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
