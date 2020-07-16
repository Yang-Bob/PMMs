import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
from models.PMMs import PMMs

# The Code of baseline network is referenced from https://github.com/icoz69/CaNet
# The code of training & testing is referenced from https://github.com/xiaomengyc/SG-One

class OneModel(nn.Module):
    def __init__(self, args):

        self.inplanes = 64
        self.num_pro = 3
        super(OneModel, self).__init__()

        self.model_res = resnet.Res50_Deeplab(pretrained=True)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer55 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer56 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer6 = ASPP.PSPnet()

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )

        self.layer9 = nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=True)  # numclass = 2

        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.PMMs = PMMs(256, self.num_pro).cuda()

        self.batch_size = args.batch_size

    def forward(self, query_rgb, support_rgb, support_mask):
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)

        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)

        # PMMs
        vec_pos, Prob_map = self.PMMs(support_feature, support_mask, query_feature)

        # feature concate
        feature_size = query_feature.shape[-2:]

        for i in range(self.num_pro):

            vec = vec_pos[i]
            exit_feat_in_ = self.f_v_concate(query_feature, vec, feature_size)
            exit_feat_in_ = self.layer55(exit_feat_in_)
            if i == 0:
                exit_feat_in = exit_feat_in_
            else:
                exit_feat_in = exit_feat_in + exit_feat_in_
        exit_feat_in = self.layer56(exit_feat_in)


        # segmentation
        out, _ = self.Segmentation(exit_feat_in, Prob_map)

        return support_feature, query_feature, vec_pos, out

    def forward_5shot(self, query_rgb, support_rgb_batch, support_mask_batch):
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # feature concate
        feature_size = query_feature.shape[-2:]

        for i in range(support_rgb_batch.shape[1]):
            support_rgb = support_rgb_batch[:, i]
            support_mask = support_mask_batch[:, i]
            # extract support feature
            support_feature = self.extract_feature_res(support_rgb)
            support_mask_temp = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear',
                                              align_corners=True)
            if i == 0:
                support_feature_all = support_feature
                support_mask_all = support_mask_temp
            else:
                support_feature_all = torch.cat([support_feature_all, support_feature], dim=2)
                support_mask_all = torch.cat([support_mask_all, support_mask_temp], dim=2)

        vec_pos, Prob_map = self.PMMs(support_feature_all, support_mask_all, query_feature)

        for i in range(self.num_pro):
            vec = vec_pos[i]
            exit_feat_in_ = self.f_v_concate(query_feature, vec, feature_size)
            exit_feat_in_ = self.layer55(exit_feat_in_)
            if i == 0:
                exit_feat_in = exit_feat_in_
            else:
                exit_feat_in = exit_feat_in + exit_feat_in_

        exit_feat_in = self.layer56(exit_feat_in)

        out, _ = self.Segmentation(exit_feat_in, Prob_map)

        return out, out, out, out

    def extract_feature_res(self, rgb):
        out_resnet = self.model_res(rgb)
        stage2_out = out_resnet[1]
        stage3_out = out_resnet[2]
        out_23 = torch.cat([stage2_out, stage3_out], dim=1)
        feature = self.layer5(out_23)

        return feature

    def f_v_concate(self, feature, vec_pos, feature_size):
        fea_pos = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        exit_feat_in = torch.cat([feature, fea_pos], dim=1)

        return exit_feat_in

    def Segmentation(self, feature, history_mask):
        feature_size = feature.shape[-2:]

        history_mask = F.interpolate(history_mask, feature_size, mode='bilinear', align_corners=True)
        out = feature
        out_plus_history = torch.cat([out, history_mask], dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer9(out)

        out_softmax = F.softmax(out, dim=1)

        return out, out_softmax

    def get_loss(self, logits, query_label, idx):
        bce_logits_func = nn.CrossEntropyLoss()
        outB, outA_pos, vec, outB_side = logits

        b, c, w, h = query_label.size()
        outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')

        # add
        query_label = query_label.view(b, -1)
        bb, cc, _, _ = outB_side.size()
        outB_side = outB_side.view(b, cc, w * h)
        #

        loss_bce_seg = bce_logits_func(outB_side, query_label.long())

        loss = loss_bce_seg

        return loss, 0, 0

    def get_pred(self, logits, query_image):
        outB, outA_pos, outB_side1, outB_side = logits
        w, h = query_image.size()[-2:]
        outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
        out_softmax = F.softmax(outB_side, dim=1)
        values, pred = torch.max(out_softmax, dim=1)
        return out_softmax, pred
