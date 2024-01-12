import contextlib
from commons import *
from utils.general import make_divisible
from detect_head import Detect

modules = [Conv, GhostConv, BottleNeck, SPP, SPPF, Focus, BottleNeckCSP,
                 C3, ShuffleBlock, conv_bn_relu_maxpool, DWConvBlock, MBConvBlock, LC3,
                 SEBlock, MobileV3Block, Hswish, SELayer, Stem, CBH, LCBlock, Dense,
                 GhostConv, ESBlottleNeck, ESSEModule]


def parse_model(net_dict, ch): # net_dict, input_channels(3)
    anchors, nc, gd, gw, act = net_dict['anchors'], \
                               net_dict['nc'],\
                               net_dict['depth_multiple'],\
                               net_dict['width_multiple'], net_dict.get('activation')
    if act:
        Conv.default_act = eval(act)
        print(f"activation: {act}")

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors # number of anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1] # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(net_dict['backbone'] + net_dict['head']): # from, number, module, args
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            # try:
            #     args[j] = eval(a) if isinstance(a, str) else a
            # except:
            #     pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n # depth gain
        # print(f"wanchao layer: {i}, {m}")
        if m in modules:

            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleNeckCSP, C3]:
                args.insert(2, n) # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is ADD:
            c2 = sum(ch[x] for x in f) // 2
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int): # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np # attach index, 'from' index, type, number params
        print(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
