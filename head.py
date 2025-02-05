
################################  LDADH  ################################

class LDADH(nn.Module):
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=512, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        self.share_conv1 = nn.ModuleList([
            Conv(hidc // 4, hidc // 2, 3),
            Conv(hidc // 2, hidc // 2, 3),
            Conv(hidc, hidc // 2, 3)
        ])
        self.share_conv2 = nn.ModuleList([
            Conv(hidc // 2, hidc // 2, 3),
            Conv(hidc // 2, hidc // 2, 3),
            Conv(hidc // 2, hidc // 2, 3)
        ])

        # self.share_conv1 = nn.ModuleList([
        #     Conv_GN(hidc // 4, hidc // 2, 3),
        #     Conv_GN(hidc // 2, hidc // 2, 3),
        #     Conv_GN(hidc, hidc // 2, 3)
        # ])
        # self.share_conv2 = nn.ModuleList([
        #     Conv_GN(hidc // 2, hidc // 2, 3),
        #     Conv_GN(hidc // 2, hidc // 2, 3),
        #     Conv_GN(hidc // 2, hidc // 2, 3)
        # ])

        self.cls_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.reg_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.DyDCNV2 = DyDCNv2(hidc // 2, hidc // 2)
        self.spatial_conv_offset = nn.Conv2d(hidc, 3 * 3 * 3, 3, padding=1)
        self.offset_dim = 2 * 3 * 3
        self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)
        self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            stack_res_list = [self.share_conv1[i](x[i])]
            stack_res_list.append(self.share_conv2[i](stack_res_list[-1]))  # 使用append而不是extend
            feat = torch.cat(stack_res_list, dim=1)

            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)

            # reg alignment
            offset_and_mask = self.spatial_conv_offset(feat)
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            reg_feat = self.DyDCNV2(reg_feat, offset, mask)

            # cls alignment
            cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()

            x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)

        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and hasattr(self, 'format') and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[:m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    
################################  LDADH  ################################