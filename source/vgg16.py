import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from vgg import vgg16
from torch.utils.tensorboard import SummaryWriter
from sklearn import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 10
data = datasets.load_digits()
class VGGBASEnet(nn.Module):
    def __init__(self):
        super(VGGBASEnet, self).__init__()
        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.fully6 = nn.Linear(1024, 128)

        self.fully7 = nn.Conv2d(128, 10)

        # Load pretrained layers
        self.load_pretrained_layers()


    def foward(self, image): #(None,3, 300, 300)

        x = F.relu(self.conv1_1(image))#(None, 64, 300, 300)
        x = F.relu(self.conv1_2(x))    #(None, 64, 300, 300 )
        x = self.pool1(x)              #(None, 64, 150, 150)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.fully6(x))
        x = (self.fully7(x))
        # Lower-level feature maps
        return x
    def load_weigt(self):
        """
             As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
             There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
             We copy these parameters into our network. It's straightforward for conv1 to conv5.
             However, the original VGG-16 does not contain the conv6 and con7 layers.
             Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
             """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
net = VGGBASEnet()
crition = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=net.parameters(), lr = 0.001)
for i in range(1, EPOCH):
