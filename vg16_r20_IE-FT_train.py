'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import torchvision
import torchvision.transforms as transforms
from model import *
import os
import argparse
import numpy as np
from models import *
from resnet_56_fea import *
# from _967_1_model import small_network
from loss import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--alpha', default=0.9, type=float, help='KD loss alpha')
parser.add_argument('--temperature', default=20, type=int, help='KD loss temperature')
# parser.add_argument('--model_name', action='_5_model',
#                     help='model name for save path')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

save_path = './checkpoint/CSVT_15_1_model'
save_path_pth = os.path.join(save_path, 'ckpt.pth')

embedT_pth = os.path.join(save_path, 'embedT_ckpt.pth')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def _load_one_loss(self, loss, args={}):
    Loss = '.'.join(['loss', loss])
    Loss = import_class(Loss)
    return Loss(**args).cuda()



# Model
print('==> Building model..')
# net = VGG('VGG16')
# net = VGG('VGG19')
# net = ResNet50()
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
# net = small_network()
# net = resnet56()
# net_teacher = VGG('VGG16')
# net_teacher = resnet56()

# teacher = resnet56()
# teacher.linear = nn.Linear(64, 100)
teacher = WRN40_2(100)
# print(teacher)
# exit()
# student = resnet20()
student = resnet20()
# student.linear = nn.Linear(64, 100)
embed_SI = auto_encoder(cin=32, cout=32)  # # 继承
embed_SE = auto_encoder(cin=32, cout=32)  # # 探索
embed_T = auto_encoder(cin=128, cout=32)  # # 教师，先训练

teacher.cuda()
student.cuda()
embed_T.cuda()
embed_SI.cuda()
embed_SE.cuda()


# net = net.cuda()
# net_teacher = net_teacher.cuda()

# checkpoint = torch.load('./checkpoint/_55_model/ckpt.pth')
# checkpoint = torch.load('./checkpoint/resnet50/ckpt.pth')
# checkpoint = torch.load('./checkpoint/vgg16/ckpt.pth')
# checkpoint = torch.load('./checkpoint/_56_model/ckpt.pth')
# checkpoint = torch.load('./checkpoint/resnet56/ckpt.pth')
checkpoint = torch.load('./checkpoint/WRN-40-2_1/ckpt.pth')

teacher.load_state_dict(checkpoint['net'])

# checkpoint_embed_T = torch.load('./checkpoint/_xiu_926_1_model/embedT_ckpt.pth')
# embed_T.load_state_dict(checkpoint_embed_T['net'])

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
bce_WL = nn.BCEWithLogitsLoss()
inh_loss = FTLoss()
exp_loss = FTLoss()
rec_loss = RecLoss()


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # alpha = params.alpha
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (0.9 * T * T) + \
              F.cross_entropy(outputs, labels) * 0.3

    return KD_loss


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret



## 可能有问题
def divide_features(feature, divide_init):
    global inh_index
    global exp_index
    # inh_index = getattr(self, 'inh_index', None)
    # exp_index = getattr(self, 'exp_index', None)
    # if inh_index is None or exp_index is None:
    #     # self.print_log('Init Inh and Exp Index')
    #     print('Init Inh and Exp Index')
    if divide_init == False:

        length = feature.size(1)
    # if self.arg.divide == 'random':
        index = torch.randperm(length).cuda()
        inh_index = index[:int(length / 2)]
        exp_index = index[int(length / 2):]

    # elif self.arg.divide == 'natural':
    #     index = torch.arange(length).cuda()
    #     self.inh_index = index[:int(length / 2)]
    #     self.exp_index = index[int(length / 2):]
    # elif self.arg.divide == 'relation':
    #     raise NotImplementedError
    # else:
    #     raise ValueError()
    # self.print_log('Inh Index: {}'.format(self.inh_index.tolist()))
    # self.print_log('Exp Index: {}'.format(self.exp_index.tolist()))
        print('Inh Index: {}'.format(inh_index.tolist()))
        print('Exp Index: {}'.format(exp_index.tolist()))
    # print('Inh Index: {}'.format(inh_index.tolist()))
    # print('Exp Index: {}'.format(exp_index.tolist()))
    inh_feature = feature.index_select(dim=1, index=inh_index)
    exp_feature = feature.index_select(dim=1, index=exp_index)
    return inh_feature, exp_feature

optimizerS = optim.SGD(student.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
schedulerS = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerS, T_max=40)
# schedulerS = torch.optim.lr_scheduler.MultiStepLR(optimizerS, milestones=[80, 120], gamma=0.1)
# schedulerS = torch.optim.lr_scheduler.MultiStepLR(optimizerS, milestones=[80, 120, 180], gamma=0.1)

optimizerT = optim.SGD(embed_T.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
schedulerT = torch.optim.lr_scheduler.MultiStepLR(optimizerT, milestones=[30, 40], gamma=1)

# Training
inh_loss_weight = 25
exp_loss_weight = 0.5
divide_init = 0

# inh_loss_weight_class = LossWeightDecay(policy='MultiStep', base=50, milestones=[80, 120])
# exp_loss_weight_class = LossWeightDecay(policy='MultiStep', base=50, milestones=[80, 120])



def train(epoch):
    print('\nEpoch: %d' % epoch)
    global divide_init
    teacher.eval()
    embed_SI.train()
    embed_SE.train()
    student.train()
    embed_T.eval()

    # inh_loss_weight = inh_loss_weight_class.step(epoch)
    # exp_loss_weight = exp_loss_weight_class.step(epoch)

    losses = [AverageMeter() for _ in range(4)]
    accs = AverageMeter()

    # train_loss = 0
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.float().cuda(), targets.long().cuda()
        optimizerS.zero_grad()

        outputsS, featuresS = student(inputs)
        if epoch == 0 and divide_init == 0:
            featuresSI, featuresSE = divide_features(featuresS, False)
            divide_init = 1

        else:
            featuresSI, featuresSE = divide_features(featuresS, True)

        factorsSI, _ = embed_SI(featuresSI)
        factorsSE, _ = embed_SE(featuresSE)

        _, featuresT = teacher(inputs)
        factorsT, _ = embed_T(featuresT)

        loss_ce = criterion(outputsS, targets)
        loss_inh = inh_loss(factorsSI, factorsT)
        loss_exp = 1 - exp_loss(factorsSE, factorsSI)
        loss = loss_ce + inh_loss_weight * loss_inh + exp_loss_weight * loss_exp

        loss.backward()
        optimizerS.step()

        prec = accuracy(outputsS, targets, topk=(1,))[0]
        losses[0].update(loss.item(), inputs.size(0))
        losses[1].update(loss_ce.item(), inputs.size(0))
        losses[2].update(loss_inh.item(), inputs.size(0))
        losses[3].update(loss_exp.item(), inputs.size(0))
        accs.update(prec.item(), inputs.size(0))

    print('Student Train total Loss: {:.2f} Acc: {:.4f}'.format(losses[0].avg, accs.avg))
    print('CE Loss: {:.2f}'.format(losses[1].avg))
    print('Inh Loss: {:.2f}'.format(losses[2].avg))
    print('Exp Loss: {:.2f}'.format(losses[3].avg))


def test(epoch):
    global best_acc

    teacher.eval()
    embed_SI.eval()
    embed_SE.eval()
    student.eval()
    embed_T.eval()

    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.float().cuda(), targets.long().cuda()
            outputs, _ = student(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = test_loss / (batch_idx + 1)
        epoch_acc = correct / total
        print('Student test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': student.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path_pth)
        best_acc = acc


def train_embed(epoch):
    # global_step += 1
    teacher.eval()
    embed_SI.eval()
    embed_SE.eval()
    student.eval()
    embed_T.train()
    losses = AverageMeter()
    print('\nEpoch: %d' % epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.float().cuda(), targets.long().cuda()
        optimizerT.zero_grad()
        outputs, features = teacher(inputs)
        # print(features.shape)
        # exit()
        _, rec_features = embed_T(features)
        loss = rec_loss(features, rec_features)
        # prec, = accuracy(outputs, targets, topk=(1,))
        loss.backward()
        optimizerT.step()
        losses.update(loss.item(), inputs.size(0))
    print('Embed_T Train Loss: {:.3f}'.format(losses.avg))


def eval_embed(epoch):
    teacher.eval()
    embed_SI.eval()
    embed_SE.eval()
    student.eval()
    embed_T.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    global best_embdT_acc
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.float().cuda(), targets.long().cuda()

            outputs, features = teacher(inputs)
            _, rec_features = embed_T(features)

            loss = rec_loss(features, rec_features)
            prec, = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            accs.update(prec.item(), inputs.size(0))
        print('embed_T test Loss: {:.4f} Acc: {:.4f}'.format(losses.avg, accs.avg))

        # ## 保存embed_T权重
        state = {
            'net': embed_T.state_dict(),
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, embedT_pth)
        print('save ! ')
    # ##


if __name__ == '__main__':

    # divide_init = False
    # embed_T.train()
    # embed_SI.train()
    # embed_SE.train()
    #
    # for epoch in range(0, 30):
    #     train_embed(epoch)
    #     eval_embed(epoch)


    for epoch in range(start_epoch, start_epoch+290):
        train(epoch)
        test(epoch)
        schedulerS.step()
