import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .spp_layer import spatial_pyramid_pool
from utils.ssw import ssw, feature_mapping
from math import floor
from torchvision.ops import roi_pool

# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
"""
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}"""
class WSDDN(nn.Module):
    def __init__(self, backbone="VGG16", pretrained=False, classes_num=20):
        super(WSDDN, self).__init__()
        # assert backbone in {"vgg"}, "`base_net` should be in {alexnet, vgg}"

        self.pretrained = pretrained
        self.classes_num = classes_num
        self.roi_output_size = (7, 7)
        
        if backbone == "VGG16":
            self.backbone = torchvision.models.vgg16(pretrained=self.pretrained)
        
        """
        self.feature = self.feature = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )"""
        self.features = self.backbone.features[:-1]
        self.fc6_and_fc7 = self.backbone.classifier[:-1]
        # self.fc6 = nn.Linear(4096, 4096)  # ???这里有个问题就是input_size
        # self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, self.classes_num)
        self.fc8d = nn.Linear(4096, self.classes_num)
    """
    def _make_layers(self, cfg):  #init VGG
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)"""
    """
    def forward(self, x, region_proposal):  # x.shape = [BATCH_SIZE, 3, h, w]  region_proposal.shape = [BATCH_SIZE, R, 4]
        x = self.feature(x)  # [BATCH_SIZE, 512, h/16, w/16]
        # print(region_proposal.shape)   
        roi = self.spp(x, region_proposal)
        x = self.fc6(roi)
        x = self.fc7(x)
        x_c = self.fc8c(x)
        x_d = self.fc8d(x)
        segma_x_c = F.softmax(x_c, dim = 2)
        segma_x_d = F.softmax(x_d, dim = 1)
        element_wise_scores = segma_x_c * segma_x_d     # x.shape = [BATCH_SIZE, R, C]
        class_result = torch.sum(element_wise_scores, dim = 1)
        return class_result"""
    def forward(self, x, region_proposal):
        # assume batch size is 1
        # print(region_proposal.shape)
        region_proposal = [region_proposal[0]]
        # print(x.shape)
        out = self.features(x)  # [1, 512, 30, 30]
        # print(out.shape)
        # print(len(region_proposal[0]))
        out = roi_pool(out, region_proposal, self.roi_output_size, 1.0 / 16)  # spp
        # print(out.shape)
        out = out.view(len(region_proposal[0]), -1)
        # print(out.shape)

        # out = out * batch_scores[0]  # apply box scores
        out = self.fc6_and_fc7(out)  # [4000, 4096]
        # print(out.shape)

        classification_scores = F.softmax(self.fc8c(out), dim=1)
        detection_scores = F.softmax(self.fc8d(out), dim=0)
        combined_scores = classification_scores * detection_scores
        # print(combined_scores.shape)
        return combined_scores
    
    @staticmethod
    def calculate_loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=0)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        return loss
    """
    def spp(self, x, region_proposal): # x.shape = [BATCH_SIZE, 512, h/16, w/16] region_proposal.shape = [BATCH_SIZE, R, 4] y.shape = [BATCH_SIZE, R, 4096]
        for i in range(1):  # BATCH_SIZE
            region_b = self.one_batch(i, x, region_proposal)
            # re = region
            # 把所有batch都拼接起来
            if i == 0:
                y = torch.unsqueeze(region_b, 0)
            else:
                y = torch.cat((y, torch.unsqueeze(region_b, 0)))
        return y

    def one_batch(self, i, x, region_proposal):
        global region
        for j in range(region_proposal.size(1)):   # R
            fmap_region = torch.unsqueeze(x[i, :, floor(region_proposal[i, j, 0]) : floor(region_proposal[i, j, 0] + region_proposal[i, j, 2]), 
                                    floor(region_proposal[i, j, 1]) : floor(region_proposal[i, j, 1] + region_proposal[i, j, 3])], 0) # 切片 [1, 512, region_height, region_width] 从feature_map中取出region_proposal
            # print("previous_conv:",fmap_region.shape) # 改的话可以在上面这里改
            f_region = spatial_pyramid_pool(previous_conv = fmap_region, num_sample = 1, 
                                    previous_conv_size = [fmap_region.size(2),fmap_region.size(3)], out_pool_size = [2, 2]) # [1, 4096]
            # print(fmap_region.shape)
            # 把1个batch的所有region_proposal都concate进来
            if j == 0:
                region = f_region
            else:
                region = torch.cat((region, f_region), 0)
                # print(region.shape)
        return region"""
"""
if __name__ == '__main__':
    BATCH_SIZE = 8
    R = 100
    net_test = WSDDN(backbone='vgg', classes_num=20)
    x_test = torch.randn(BATCH_SIZE, 3, 512, 512)
    ssw_spp = torch.zeros(BATCH_SIZE, R, 4)

    for i in range(BATCH_SIZE):
        for j in range(R):
            ssw_spp[i, j, 0] = 2
            ssw_spp[i, j, 1] = 2
            ssw_spp[i, j, 2] = 8  # height
            ssw_spp[i, j, 3] = 6  # width
    out_test = net_test(x_test, ssw_spp)
    print(out_test.shape)"""
    
