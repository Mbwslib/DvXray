import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.relu(x + 3) / 6

class r_func(nn.Module):

    def __init__(self, int_c, out_c, reduction=32):

        super(r_func, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((1, None))

        def _make_basic(input_dim, output_dim, kernel_size, stride, padding):

            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                          padding),
                nn.BatchNorm2d(output_dim))
        # self.act = h_swish()
        self.act = nn.ReLU6(inplace=True)
        self.dcov = _make_basic(int_c, (out_c // reduction), kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv2d((out_c // reduction), out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, h_x, l_x, sig):

        B, _, H, W = l_x.size()
        m_x = self.pool_h(h_x)
        m_x = F.interpolate(m_x, size=(1, W), mode='bilinear')
        m_x = self.act(self.dcov(m_x))
        m_out = l_x * (self.conv_h(m_x).sigmoid())
        sig = sig.reshape((B, 1, 1, 1))
        out = l_x + sig * m_out

        return out



class AHCR(nn.Module):

    def __init__(self, num_classes=15):

        super(AHCR, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for item in resnet.children():
            if isinstance(item, nn.BatchNorm2d):
                item.affine = False

        self.features = nn.Sequential(resnet.conv1,
                                      resnet.bn1,
                                      resnet.relu,
                                      resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cov4_ol = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)
        self.cov4_sd = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)

        self.cov3_ol = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.cov3_sd = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)

        self.cov2_ol = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.cov2_sd = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        self.ol_r_sd_1 = r_func(2048, 1024)
        self.sd_r_ol_1 = r_func(2048, 1024)

        self.ol_r_sd_2 = r_func(1024, 512)
        self.sd_r_ol_2 = r_func(1024, 512)

        self.po1 = nn.AvgPool2d(7, stride=1)
        self.po2 = nn.AvgPool2d(14, stride=1)
        self.po3 = nn.AvgPool2d(28, stride=1)

        self.fc1_ol = nn.Linear(2048, num_classes)
        self.fc1_sd = nn.Linear(2048, num_classes)

        self.fc2_ol = nn.Linear(1024, num_classes)
        self.fc2_sd = nn.Linear(1024, num_classes)

        self.fc3_ol = nn.Linear(512, num_classes)
        self.fc3_sd = nn.Linear(512, num_classes)

        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)

        self.init_weights()

    def init_weights(self):

        for name, param in self.named_parameters():

            if name in ('cov4_ol.weight', 'cov4_sd.weight', 'cov3_ol.weight', 'cov3_sd.weight', 'cov2_ol.weight',
                        'cov2_sd.weight', 'ol_r_sd_1.dcov.0.weight', 'ol_r_sd_1.conv_h.weight', 'sd_r_ol_1.dcov.0.weight', 'sd_r_ol_1.conv_h.weight',
                        'ol_r_sd_2.dcov.0.weight', 'ol_r_sd_2.conv_h.weight', 'sd_r_ol_2.dcov.0.weight', 'sd_r_ol_2.conv_h.weight'):
                nn.init.orthogonal_(param)
            if name in ('cov4_ol.bias', 'cov4_sd.bias', 'cov3_ol.bias', 'cov3_sd.bias', 'cov2_ol.bias',
                        'cov2_sd.bias', 'ol_r_sd_1.dcov.0.bias', 'ol_r_sd_1.conv_h.bias', 'sd_r_ol_1.dcov.0.bias', 'sd_r_ol_1.conv_h.bias',
                        'ol_r_sd_2.dcov.0.bias', 'ol_r_sd_2.conv_h.bias', 'sd_r_ol_2.dcov.0.bias', 'sd_r_ol_2.conv_h.bias'):
                nn.init.constant_(param, 0.0)

            if name in ('fc1_ol.weight', 'fc1_sd.weight', 'fc2_ol.weight', 'fc2_sd.weight', 'fc3_ol.weight', 'fc3_sd.weight'):
                nn.init.normal_(param, 0, 0.01)
            if name in ('fc1_ol.bias', 'fc1_sd.bias', 'fc2_ol.bias', 'fc2_sd.bias', 'fc3_ol.bias', 'fc3_sd.bias'):
                nn.init.zeros_(param)

    def forward(self, image_ol, image_sd):

        bf_ol = self.features(image_ol)
        f_l1_o1 = self.layer1(bf_ol)
        f_l2_ol = self.layer2(f_l1_o1)
        f_l3_ol = self.layer3(f_l2_ol)
        f_l4_ol = self.layer4(f_l3_ol)

        bf_sd = self.features(image_sd)
        f_l1_sd = self.layer1(bf_sd)
        f_l2_sd = self.layer2(f_l1_sd)
        f_l3_sd = self.layer3(f_l2_sd)
        f_l4_sd = self.layer4(f_l3_sd)

        out1_ol = self.fc1_ol(self.po1(F.relu(self.cov4_ol(f_l4_ol))).view(f_l4_ol.size(0), -1))
        out1_sd = self.fc1_sd(self.po1(F.relu(self.cov4_sd(f_l4_sd))).view(f_l4_sd.size(0), -1))

        con_coe1 = F.relu(self.cos_similarity(out1_ol, out1_sd))

        out2_m_ol = self.cov3_ol(self.sd_r_ol_1(f_l4_sd, f_l3_ol, con_coe1))
        out2_m_sd = self.cov3_sd(self.ol_r_sd_1(f_l4_ol, f_l3_sd, con_coe1))

        out2_ol = self.fc2_ol(self.po2(F.relu(out2_m_ol)).view(out2_m_ol.size(0), -1))
        out2_sd = self.fc2_sd(self.po2(F.relu(out2_m_sd)).view(out2_m_sd.size(0), -1))

        con_coe2 = F.relu(self.cos_similarity(out2_ol, out2_sd))

        out3_m_ol = self.cov2_ol(self.sd_r_ol_2(out2_m_sd, f_l2_ol, con_coe2))
        out3_m_sd = self.cov2_sd(self.ol_r_sd_2(out2_m_ol, f_l2_sd, con_coe2))

        out3_ol = self.fc3_ol(self.po3(F.relu(out3_m_ol)).view(out3_m_ol.size(0), -1))
        out3_sd = self.fc3_sd(self.po3(F.relu(out3_m_sd)).view(out3_m_sd.size(0), -1))

        ol_output = (out1_ol + out2_ol + out3_ol) / 3.

        sd_output = (out1_sd + out2_sd + out3_sd) / 3.

        return ol_output, sd_output

if __name__ == '__main__':

    from thop import profile
    from thop import clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AHCR(num_classes=15).to(device)

    flops, params = profile(model, (torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 224, 224).to(device)))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops)
    print(params)


