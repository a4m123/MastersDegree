import torch
import torch.nn as nn

class UNet(nn.Module): #markovng
    def __init__(self, in_classes = 3, out_classes = 5, kernel_size_temp = 3, dropout=0.125):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(in_classes, 64, kernel_size=kernel_size_temp, padding=1)
        self.elu1 = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size_temp, padding=1)
        self.elu2 = nn.ELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=dropout)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size_temp, padding=1)
        self.elu3 = nn.ELU(inplace=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(p=dropout)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=kernel_size_temp, padding=1)
        self.elu4 = nn.ELU(inplace=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout2d(p=dropout)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=kernel_size_temp, padding=1)
        self.elu5 = nn.ELU(inplace=True)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout5 = nn.Dropout2d(p=dropout)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=kernel_size_temp, padding=1)
        self.elu6 = nn.ELU(inplace=True)
        self.bn6 = nn.BatchNorm2d(256)
        self.dropout6 = nn.Dropout2d(p=dropout)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=kernel_size_temp, padding=1)
        self.elu7 = nn.ELU(inplace=True)
        self.bn7 = nn.BatchNorm2d(512)
        self.dropout7 = nn.Dropout2d(p=dropout)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=kernel_size_temp, padding=1)
        self.elu8 = nn.ELU(inplace=True)
        self.bn8 = nn.BatchNorm2d(512)
        self.dropout8 = nn.Dropout2d(p=dropout)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=kernel_size_temp, padding=1)
        self.elu9 = nn.ELU(inplace=True)
        self.bn9 = nn.BatchNorm2d(1024)
        self.dropout9 = nn.Dropout2d(p=dropout)

        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=kernel_size_temp, padding=1)
        self.elu10 = nn.ELU(inplace=True)
        self.bn10 = nn.BatchNorm2d(1024)
        self.dropout10 = nn.Dropout2d(p=dropout)

        # Expanding path

        self.upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=kernel_size_temp, padding=1)
        self.elu11 = nn.ELU(inplace=True)
        self.bn11 = nn.BatchNorm2d(512)
        self.dropout11 = nn.Dropout2d(p=dropout)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=kernel_size_temp, padding=1)
        self.elu12 = nn.ELU(inplace=True)
        self.bn12 = nn.BatchNorm2d(512)
        self.dropout12 = nn.Dropout2d(p=dropout)

        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=kernel_size_temp, padding=1)
        self.elu13 = nn.ELU(inplace=True)
        self.bn13 = nn.BatchNorm2d(256)
        self.dropout13 = nn.Dropout2d(p=dropout)

        self.conv14 = nn.Conv2d(256, 256, kernel_size=kernel_size_temp, padding=1)
        self.elu14 = nn.ELU(inplace=True)
        self.bn14 = nn.BatchNorm2d(256)
        self.dropout14 = nn.Dropout2d(p=dropout)

        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(256, 128, kernel_size=kernel_size_temp, padding=1)
        self.elu15 = nn.ELU(inplace=True)
        self.bn15 = nn.BatchNorm2d(128)
        self.dropout15 = nn.Dropout2d(p=dropout)

        self.conv16 = nn.Conv2d(128, 128, kernel_size=kernel_size_temp, padding=1)
        self.elu16 = nn.ELU(inplace=True)
        self.bn16 = nn.BatchNorm2d(128)
        self.dropout16 = nn.Dropout2d(p=dropout)

        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv17 = nn.Conv2d(128, 64, kernel_size=kernel_size_temp, padding=1)
        self.elu17 = nn.ELU(inplace=True)
        self.bn17 = nn.BatchNorm2d(64)
        self.dropout17 = nn.Dropout2d(p=dropout)

        self.conv18 = nn.Conv2d(64, 64, kernel_size=kernel_size_temp, padding=1)
        self.elu18 = nn.ELU(inplace=True)
        self.bn18 = nn.BatchNorm2d(64)
        self.dropout18 = nn.Dropout2d(p=dropout)

        self.conv19 = nn.Conv2d(64, out_classes, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.input(x)
        # Contracting path
        x = self.conv1(x)
        x = self.elu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.elu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x1 = x
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.elu3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.elu4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        x2 = x
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.elu5(x)
        x = self.bn5(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.elu6(x)
        x = self.bn6(x)
        x = self.dropout6(x)

        x3 = x
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.elu7(x)
        x = self.bn7(x)
        x = self.dropout7(x)

        x = self.conv8(x)
        x = self.elu8(x)
        x = self.bn8(x)
        x = self.dropout8(x)

        x4 = x
        x = self.pool4(x)

        x = self.conv9(x)
        x = self.elu9(x)
        x = self.bn9(x)
        x = self.dropout9(x)

        x = self.conv10(x)
        x = self.elu10(x)
        x = self.bn10(x)
        x = self.dropout10(x)

        # Expanding path

        x = self.upsample1(x)
        x = torch.cat([x, x4], dim=1)

        x = self.conv11(x)
        x = self.elu11(x)
        x = self.bn11(x)
        x = self.dropout11(x)

        x = self.conv12(x)
        x = self.elu12(x)
        x = self.bn12(x)
        x = self.dropout12(x)

        x = self.upsample2(x)
        x = torch.cat([x, x3], dim=1)

        x = self.conv13(x)
        x = self.elu13(x)
        x = self.bn13(x)
        x = self.dropout13(x)

        x = self.conv14(x)
        x = self.elu14(x)
        x = self.bn14(x)
        x = self.dropout14(x)

        x = self.upsample3(x)
        x = torch.cat([x, x2], dim=1)

        x = self.conv15(x)
        x = self.elu15(x)
        x = self.bn15(x)
        x = self.dropout15(x)

        x = self.conv16(x)
        x = self.elu16(x)
        x = self.bn16(x)
        x = self.dropout16(x)

        x = self.upsample4(x)
        x = torch.cat([x, x1], dim=1)

        x = self.conv17(x)
        x = self.elu17(x)
        x = self.bn17(x)
        x = self.dropout17(x)

        x = self.conv18(x)
        x = self.elu18(x)
        x = self.bn18(x)
        x = self.dropout18(x)

        x = self.conv19(x)
        x = self.softmax(x)

        return x

    def predict(self, x):
        pred = self.forward(x)
        pred = pred.argmax(dim=1)
        return pred
