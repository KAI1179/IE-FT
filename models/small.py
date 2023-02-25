import torch
import torchvision
import torch.nn as nn
from torch.nn import init

from block_base import block_base
from block_mid import block_mid_1, block_mid_2
from block_group import block_group


load_path = '/home/xuk/桌面/git_code/pytorch-cifar-master/checkpoint/vgg16_bn-6c64b313.pth'
pretrained_dict = torch.load(load_path)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNRelu, self).__init__()

        self.conbnrelu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )

        for m in self.conbnrelu.children():
            m.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conbnrelu(x)

        return x


class classifier(nn.Module):
    def __init__(self, in_features, class_num):
        super(classifier, self).__init__()

        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features=in_features, out_features=class_num),
        )

        for m in self.classifier.children():
            m.apply(weights_init_classifier)

    def forward(self, x):
        x = self.classifier(x)

        return x


class small_network(nn.Module):
    def __init__(self, class_num=10):
        super(small_network, self).__init__()
        self.class_num = class_num

        self.base = block_base(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool_base = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.base.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.base.load_state_dict(model_dict)
        #  #load weight**************************

        # #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓task-1↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓##
        self.block_mid_1_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_1_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_1_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_1_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_1_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_1_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_1_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_1_2.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_group_1 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        #  #load weight**************************
        model_dict = self.block_group_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_1.load_state_dict(model_dict)
        #  #load weight**************************
        ##↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑task-1↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑##


        # #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓task-2↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓##
        self.block_mid_2_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_2_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_2_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_2_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_2_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_2_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_2_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_2_2.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_group_2 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        #  #load weight**************************
        model_dict = self.block_group_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_2.load_state_dict(model_dict)
        #  #load weight**************************
        ##↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑task-2↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑##

        # #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓task-3↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓##
        self.block_mid_3_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_3_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_3_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_3_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_3_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_3_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_3_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_3_2.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_group_3 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        #  #load weight**************************
        model_dict = self.block_group_3.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_3.load_state_dict(model_dict)
        #  #load weight**************************
        ##↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑task-3↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑##

        self.block_group_1_classifier = nn.Sequential(  # group_1 :plane bird cat dog
            classifier(in_features=2048, class_num=4),
        )
        self.block_group_2_classifier = nn.Sequential(  # group_2： car truck ship
            classifier(in_features=2048, class_num=3),
        )
        self.block_group_3_classifier = nn.Sequential(  # group_3： horse deer frog
            classifier(in_features=2048, class_num=3),
        )



    def forward(self, x):
        x = self.base(x)
        x = self.pool_base(x)

        x_1 = self.block_mid_1_1(x)
        x_1 = self.pool_mid_1_1(x_1)
        x_1 = self.block_mid_1_2(x_1)
        x_1 = self.pool_mid_1_2(x_1)
        x_1 = self.block_group_1(x_1)
        x_1 = x_1.view(x_1.size(0), -1)
        x_1_out = self.block_group_1_classifier(x_1)

        x_2 = self.block_mid_2_1(x)
        x_2 = self.pool_mid_2_1(x_2)
        x_2 = self.block_mid_2_2(x_2)
        x_2 = self.pool_mid_2_2(x_2)
        x_2 = self.block_group_2(x_2)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2_out = self.block_group_2_classifier(x_2)

        x_3 = self.block_mid_3_1(x)
        x_3 = self.pool_mid_3_1(x_3)
        x_3 = self.block_mid_3_2(x_3)
        x_3 = self.pool_mid_3_2(x_3)
        x_3 = self.block_group_3(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3_out = self.block_group_3_classifier(x_3)

        preds = []
        preds.append(x_1_out[:, 0].reshape([-1, 1]))  # plane
        preds.append(x_2_out[:, 0].reshape([-1, 1]))  # car
        preds.append(x_1_out[:, 1].reshape([-1, 1]))  # bird
        preds.append(x_1_out[:, 2].reshape([-1, 1]))  # cat
        preds.append(x_3_out[:, 0].reshape([-1, 1]))  # deer
        preds.append(x_1_out[:, 3].reshape([-1, 1]))  # dog
        preds.append(x_3_out[:, 1].reshape([-1, 1]))  # frog
        preds.append(x_3_out[:, 2].reshape([-1, 1]))  # horse
        preds.append(x_2_out[:, 1].reshape([-1, 1]))  # ship
        preds.append(x_2_out[:, 2].reshape([-1, 1]))  # truck

        preds = torch.cat(preds, dim=1)

        return preds



if __name__ == '__main__':
    model = small_network()
    print(model)
    x = torch.randn([1, 3, 32, 32])
    # x_1, x_2, x_3_1, x_3_2, x_3_3, x_4 = model(x)
    # print(x_1.shape, x_2.shape, x_3_1.shape, x_3_2.shape, x_3_3.shape, x_4.shape)
    y = model(x)
    print(y.shape)
    # g = make_dot(y)
    # g.render('models_small_full', view=False)






