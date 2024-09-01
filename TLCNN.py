import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

#defining pre-trained models path
model_path = "./models/scratch_v2.pt"


# defining proposed cnn model
class TLCnnCam(nn.Module):
    def __init__(self, num_classes):
        super(TLCnnCam, self).__init__()
        # block-1
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block-2
        self.conv_layer21 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block-3
        self.conv_layer31 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer32 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(128*5*5, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

        # Store feature maps and gradients here
        self.feature_maps = {}
        self.gradients = {}

        # Register hooks for layers of interest
        self.conv_layer1.register_forward_hook(self.save_feature_maps('block1'))
        self.conv_layer21.register_forward_hook(self.save_feature_maps('block2_conv1'))
        self.conv_layer22.register_forward_hook(self.save_feature_maps('block2_conv2'))
        self.conv_layer31.register_forward_hook(self.save_feature_maps('block3_conv1'))
        self.conv_layer32.register_forward_hook(self.save_feature_maps('block3_conv2'))

        self.conv_layer1.register_backward_hook(self.save_gradients('block1'))
        self.conv_layer21.register_backward_hook(self.save_gradients('block2_conv1'))
        self.conv_layer22.register_backward_hook(self.save_gradients('block2_conv2'))
        self.conv_layer31.register_backward_hook(self.save_gradients('block3_conv1'))
        self.conv_layer32.register_backward_hook(self.save_gradients('block3_conv2'))

    def save_feature_maps(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook

    def save_gradients(self, name):
        def hook(module, grad_in, grad_out):
            self.gradients[name] = grad_out[0]
        return hook

    def forward(self, x):
        # block-1
        out = self.max_pool1(self.conv_layer1(x))

        # block-2
        out = self.max_pool2(self.conv_layer21(out))
        out = self.max_pool3(self.conv_layer22(out))

        # block-3
        out = self.max_pool4(self.conv_layer31(out))
        out = self.max_pool5(self.conv_layer32(out))

        # dropout
        out = self.dropout(out)

        # flatten
        out = out.reshape(out.size(0), -1)

        # fully connected layers
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out


# defining cam model
CamModel = TLCnnCam(num_classes=10)

if  torch.cuda.is_available():
  CamModel.load_state_dict(torch.load(model_path))
  # model.to(device)
else:
  CamModel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# evaluation mode
CamModel.eval()
