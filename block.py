
################################  RGHBlock  ################################
class Conv(nn.Module):

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))

class RGConv(nn.Module):

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):

        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = RepConv(c2//2, c2//2, k, g=(c2//2), act=act)
        self.conv3 = GhostConv(c2//2, c2//2, k , act=act)

    def forward(self, x):

        x = self.conv1(x)
        x1, x2 = x.chunk(2, 1)
        x1 = self.conv2(x1)
        x2 = self.conv3(x2)
        x_out = torch.cat([x1, x2], dim=1)
        return x_out

class RGHBlock(nn.Module):
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=True):

        super().__init__()
        block = RGConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):

        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

################################  RGHBlock  ################################

################################  RFCE  ################################

class DualConvMerge(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):

        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, k+2, 1, None, c_, act=act)

    def forward(self, x):

        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class DBConvFusion(nn.Module):

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = DualConvMerge(c1, c2, k=k, s=s, g=g, act=True)
        self.conv2 = DualConvMerge(c1, c2, k=k-2, s=s, g=g, act=True)

    def forward_fuse(self, x):

        return self.act(self.conv(x))

    def forward(self, x):

        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):

        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):

        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):

        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):

        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class RFCE(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RFCE, self).__init__()

        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)

        self.cv3 = DBConvFusion(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 3)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

################################  RFCE  ################################


################################  TFE  ################################

'Zoom_cat ---> TFE'
class TFE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

################################  TFE  ################################

################################  SSFF  ################################

'ScalSeq ---> SSFF'
class SSFF(nn.Module):
    def __init__(self, inc, channel):
        super(SSFF, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
    
################################  SSFF  ################################

################################  HGStem  ################################

class HGStem(nn.Module):

    def __init__(self, c1, cm, c2):

        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):

        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x
    
################################  HGStem  ################################

################################  DWConv  ################################

class Conv(nn.Module):

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))

class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation

        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

################################  DWConv  ################################

