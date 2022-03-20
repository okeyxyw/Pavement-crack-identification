import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
import torchvision
import os, sys
import cv2 as cv
from torch.utils.data import DataLoader, sampler
class UNetModel(torch.nn.Module):
    def __init__(self, in_features=1, out_features=2, init_features=32):
        super(UNetModel, self).__init__()
        features = init_features
        self.encode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_decode_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU()
        )
        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        enc1 = self.encode_layer1(x)
        enc2 = self.encode_layer2(self.pool1(enc1))
        enc3 = self.encode_layer3(self.pool2(enc2))
        enc4 = self.encode_layer4(self.pool3(enc3))
        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decode_layer4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decode_layer1(dec1)
        out = self.out_layer(dec1)
        return out
def test(unet):
    unet.load_state_dict(torch.load('unet_road_model.pt'))
    root_dir = 'CrackForest-dataset-master/CrackForest-dataset-master/test/'
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)
        h, w = image.shape
        img = np.float32(image) /255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view( 1, 1, h, w)
        probs = unet(x_input)#.cuda()
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        grad, output = m_label_out_.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()
        cv.imshow("input", image)
        result = cv.resize(np.uint8(predic_), (w, h))
        cv.imshow("unet-segmentation-demo", result)
        cv.waitKey(200)
    cv.destroyAllWindows()
unet=UNetModel()
test(unet)

