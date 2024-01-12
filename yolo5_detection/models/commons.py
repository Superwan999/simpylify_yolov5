import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, is_bn=True):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, padding=autopad(k, p), groups=g, bias=False),
            nn.BatchNorm2d(c2) if is_bn else nn.Identity(),
            nn.SiLU() if act is True else
            (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.conv(x)


class BottleNeck(nn.Module):
    # Standard BottleNeck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(BottleNeck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ContextBlock2d(nn.Module):
    def __init__(self, inplanes, pool='att', fusions=('channel_add', 'channel_mul')):
        """
        ContextBlock2d

        parameters:
        inplances: int, Number of in_channels.
        pool: string, spatial att or global pooling (default: 'att')
        fusion: list
        """
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = inplanes // 4
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = self.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x

        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class BottleNeckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleNeckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[BottleNeck(c_, c_, shortcut, g, e=1.0)
                                 for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        y = torch.concat([y1, y2], dim=1)
        y = self.bn(y)
        y = self.act(y)
        y = self.cv4(y)
        return y


class C3(nn.Module):
    # CSP BottleNeck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[BottleNeck(c_, c_, shortcut, g, e=1.0)
                                 for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        y = torch.concat([y1, y2], dim=1)
        y = self.cv3(y)
        return y


class C3GC(nn.Module):
    # CSP BottleNeck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3GC, self).__init__()
        c_ = int(c2 * e)
        self.gc = ContextBlock2d(c1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[BottleNeck(c_, c_, shortcut, g, e=1.0)
                                 for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(self.gc(x))
        y = torch.concat([y1, y2], dim=1)
        y = self.cv3(y)
        return y


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(3, 5, 7)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        out = torch.cat([x, y1, y2, self.m(y2)], dim=1)
        out = self.cv2(out)
        return out


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        y = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        y = self.conv(y)
        return y


class Contract(nn.Module):
    def __init__(self, gain=2):
        super(Contract, self).__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):
    def __init__(self, gain=2):
        super(Expand, self).__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s ** 2, H * s, W * s)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.d)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            Hswish()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def channel_shuffle(x, groups):
    n, c, h, w = x.data.size()
    c_per_group = c // groups

    # reshape
    x = x.view(n, groups, c_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(n, -1, h, w)

    return x


class ShuffleBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(ShuffleBlock, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError('illlegal stride value')
        self.stride = stride

        branch_features = c_out // 2
        assert (self.stride != 1) or (c_in == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(c_in, c_in, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(c_in if self.stride > 1 else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        out = channel_shuffle(out, 2)
        return out


class conv_bn_relu_maxpool(nn.Module):
    def __init__(self, c1, c2):
        super(conv_bn_relu_maxpool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class DWConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, s):
        super(DWConvBlock, self).__init__()
        self.p = k // 2
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=k,
                               stride=s, padding=self.p, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(Stem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=c2, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class MBConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, s):
        super(MBConvBlock, self).__init__()
        self._momentum = 0.01
        self._epsilon = 1e-3
        self.c_in = c_in
        self.c_out = c_out
        self.stride = s
        self.id_skip = True
        self._dw_conv = nn.Conv2d(c_in, c_in, kernel_size=k,
                                  padding=(k - 1) // 2, stride=s,
                                  groups=c_in, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=c_in, momentum=self._momentum, eps=self._epsilon
        )
        self._proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=c_out, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop_connect_rate (float, between 0 and 1)
        :return: output of block
        """
        identity = x
        x = self._relu(self._bn1(self._dw_conv(x)))
        x = self._bn2(self._proj(x))
        if (
                self.id_skip
                and self.stride == 1
                and self.c_in == self.c_out
        ):
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x = x + identity
        return x


class LC3(nn.Module):
    # CSP BottleNeck with 3 convolutions
    def __init__(self, c_in, c_out, n=1, shortcut=True, g=1, e=0.5):
        super(LC3, self).__init__()
        c_hid = int(c_out * e)
        self.conv1 = Conv(c_in, c_hid, 1, 1)
        self.conv2 = Conv(c_in, c_hid, 1, 1)
        self.conv3 = Conv(c_hid, c_out, 1)
        self.m = nn.Sequential(*[BottleNeck(c_hid, c_hid, shortcut, g, e=1.0)
                                 for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.conv1(x))
        y2 = self.conv2(x)
        y = torch.add(y1, y2)
        y = self.conv3(y)
        return y


class ADD(nn.Module):
    def __init__(self, alpha=0.5):
        super(ADD, self).__init__()
        self.a = alpha

    def forward(self, x):
        x1, x2 = x[0], x[1]
        return torch.add(x1, x2, alpha=self.a)


class SEBlock(nn.Module):
    def __init__(self, c_in, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(c_in, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(internal_neurons, c_in, kernel_size=1, stride=1, bias=True)
        self.c_in = c_in

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size(3))
        y = self.down(y)
        y = F.relu(y)
        y = self.up(y)
        y = torch.sigmoid(y)
        y = x.view(-1, self.c_in, 1, 1)
        return x * y


class MobileV3Block(nn.Module):
    def __init__(self, c_in, c_out, c_hid, kernel_size, stride, use_se, use_hs):
        super(MobileV3Block, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and c_in == c_out
        if c_in == c_hid:
            self.conv = nn.Sequential(
                nn.Conv2d(c_hid, c_hid, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2,
                          groups=c_hid, bias=False),
                nn.BatchNorm2d(c_hid),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                SELayer(c_hid) if use_se else nn.Identity(),
                nn.Conv2d(c_hid, c_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_hid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_hid),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(c_hid, c_hid, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=c_hid, bias=False),
                nn.BatchNorm2d(c_hid),
                SELayer(c_hid) if use_se else nn.Identity(),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(c_hid, c_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


class CBH(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, groups=1):
        super(CBH, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                              stride=stride, padding=(kernel_size - 1) // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class LCSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(LCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.silu(x)
        out = identity * x
        return out


class LCBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dw_size, use_se=False):
        super(LCBlock, self).__init__()
        self.use_se = use_se
        self.dw_conv = CBH(c_in, c_out, dw_size, stride, groups=c_in)
        if use_se:
            self.se = LCSEModule(c_in)
        self.pw_conv = CBH(c_in, 1, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class Dense(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dropout_prob):
        super(Dense, self).__init__()
        self.dense = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.dense(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x


class GhostConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, groups=1, act=True):
        super(GhostConv, self).__init__()
        c_hid = c_out // 2
        self.conv1 = Conv(c_in, c_hid, 1, stride, None, groups, act)
        self.conv2 = Conv(c_hid, c_hid, kernel_size, stride, None, c_hid, act=act)

    def forward(self, x):
        y = self.conv1(x)
        out = torch.cat([y, self.conv2(y)], dim=1)
        return out


class ESSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ESSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0)
        self.hash = nn.Hardswish()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hash(x)
        out = identity * x
        return out


def depthwise_conv(i, o, kernel_size=3, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


class ESBlottleNeck(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(ESBlottleNeck, self).__init__()

        assert stride in [1, 2], "illegal stride value"
        self.stride = stride
        branch_features = c_out // 2
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                depthwise_conv(c_in, c_out, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Hardswish(inplace=True)
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(c_in if self.stride > 1 else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
            depthwise_conv(branch_features, branch_features, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(branch_features),
            ESSEModule(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True)
        )

        self.branch3 = nn.Sequential(
            GhostConv(branch_features, branch_features, 3, 1),
            ESSEModule(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True)
        )
        self.branch4 = nn.Sequential(
            depthwise_conv(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x3 = torch.cat([x1, self.branch3(x2)], dim=1)
            out = channel_shuffle(x3, 2)
        else:
            x1 = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
            out = self.branch4(x1)
        return out
