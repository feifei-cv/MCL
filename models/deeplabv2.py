import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import warnings
from torchvision import models

affine_par = True

# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from models.daformer.utils import resize
from models.daformer.aspp_head import ASPPWrapper
from models.daformer.mlp import MLP

from .backbone import get_mit_backbone


######### For ResNet101 backbone
def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_target=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None, num_target=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine=affine_par)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class SEBlock(nn.Module):
    def __init__(self, inplanes, r=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
            nn.Linear(inplanes, inplanes // r),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes // r, inplanes),
            nn.Sigmoid()
        )

    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate=0.1, use_se=True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
            nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                nn.ReLU(inplace=True)]))

        for dilation, padding in zip(dilation_series, padding_series):
            # self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                    # nn.ReflectionPad2d(padding),
                    nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                    nn.ReLU(inplace=True)]))

        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                                              nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1,
                                                        padding=1, dilation=1, bias=True),
                                              nn.GroupNorm(num_groups=32, num_channels=256, affine=True)])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1,
                          bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine=True)])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
                                    nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False)])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m,
                                                                                                 nn.GroupNorm) or isinstance(
                    m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat((out, self.conv2d_list[i + 1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out


class ResNet101(nn.Module):
    def __init__(self, block, layers, num_classes, BatchNorm, num_target=1, bn_clr=False, stage=None):

        self.inplanes = 64
        self.bn_clr = bn_clr
        self.num_target = num_target
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm, num_target=num_target)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm, num_target=num_target)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, BatchNorm=BatchNorm,
                                       num_target=num_target)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, BatchNorm=BatchNorm,
                                       num_target=num_target)
        # self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        # self.layer5 = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer5_list = nn.ModuleList()
        if stage == 'stage1':
            for i in range(self.num_target):
                if i != self.num_target - 1:
                    layer = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24],
                                                  num_classes)
                else:
                    layer = self._make_pred_layer(Classifier_Module2, 2048 * (self.num_target - 2), [6, 12, 18, 24],
                                                  [6, 12, 18, 24], num_classes)  ## ensemble
                self.layer5_list.append(layer)
        else:
            for i in range(self.num_target):
                layer = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
                self.layer5_list.append(layer)

        if self.bn_clr:
            self.bn_pretrain = BatchNorm(2048, affine=affine_par)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None, num_target=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion, affine=affine_par))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, BatchNorm=BatchNorm,
                  num_target=num_target))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm, num_target=num_target))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, domain_list, ssl, target_ensembel=False, ensembel=False):

        if ensembel:
            if target_ensembel:
                x = torch.concat(ssl, dim=1)  # torch.concat(torch.split(x, b//(self.num_target-2), 0), dim=1)
            else:
                x = torch.concat(ssl[:-1], dim=1)  ## the last feature is domain-agnostic
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            ssl.append(x)
            if self.bn_clr:
                x = self.bn_pretrain(x)

        outputs = []
        for i in domain_list:
            x1 = self.layer5_list[i](x, get_feat=True)  # produce segmap
            outputs.append(x1)

        return outputs

    def get_1x_lr_params(self):

        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())

            # b.append(self.layer5.parameters())
        for layer in self.layer5_list:
            b.append(layer.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_1x_lr_params_new(self):

        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
        b1 = []
        if self.bn_clr:
            b1.append(self.bn_pretrain.parameters())

        for layer in self.layer5_list:
            if layer != self.layer5_list[-2]:  ## -1
                b1.append(layer.parameters())

        for j in range(len(b1)):
            for i in b1[j]:
                yield i

    def get_10x_lr_params_new(self):

        b = []

        # b.append(self.layer5.parameters())
        layer = self.layer5_list[-2]  ##-1
        b.append(layer.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters_new(self, args):
        return [{'params': self.get_1x_lr_params_new(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params_new(), 'lr': 10 * args.learning_rate}]

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss


def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d) \
            or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def Deeplab(BatchNorm, num_classes=7, num_target=1, freeze_bn=False, restore_from=None, initialization=None,
            bn_clr=False, stage=None):
    model = ResNet101(Bottleneck, [3, 4, 23, 3], num_classes, BatchNorm, num_target=num_target, bn_clr=bn_clr,
                      stage=stage)
    if freeze_bn:
        model.apply(freeze_bn_func)
    if initialization is None:
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    else:
        pretrain_dict = torch.load(initialization)['state_dict']
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)

    if restore_from is not None:
        checkpoint = torch.load(restore_from)['ResNet101']["model_state"]
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    return model


######### For VGG backbone
class VGG(nn.Module):

    def __init__(self, vgg, num_classes, BatchNorm, num_target=1, bn_clr=False, stage=None):
        super(VGG, self).__init__()
        self.bn_clr = bn_clr
        self.num_target = num_target

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features = nn.Sequential(*(features[i] for i in list(range(23)) + list(range(24, 30))))
        for i in [23, 25, 27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)
        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)
        self.features = nn.Sequential(
            *([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.layer5_list = nn.ModuleList()
        if stage == 'stage1':
            for i in range(self.num_target):
                if i != self.num_target - 1:
                    layer = self._make_pred_layer(Classifier_Module2, 1024, [6, 12, 18, 24], [6, 12, 18, 24],
                                                  num_classes)
                else:
                    layer = self._make_pred_layer(Classifier_Module2, 1024 * (self.num_target - 2), [6, 12, 18, 24],
                                                  [6, 12, 18, 24], num_classes)  ## ensemble
                self.layer5_list.append(layer)
        else:
            for i in range(self.num_target):
                layer = self._make_pred_layer(Classifier_Module2, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
                self.layer5_list.append(layer)

        if self.bn_clr:
            self.bn_pretrain = BatchNorm(1024, affine=affine_par)
        self._initialize_weights()

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, domain_list, ssl, target_ensembel=False, ensembel=False):

        if ensembel:
            if target_ensembel:
                x = torch.concat(ssl, dim=1)  # torch.concat(torch.split(x, b//(self.num_target-2), 0), dim=1)
            else:
                x = torch.concat(ssl[:-1], dim=1)  ## the last feature is domain-agnostic
        else:
            x = self.features(x)
            ssl.append(x)
            if self.bn_clr:
                x = self.bn_pretrain(x)

        outputs = []
        for i in domain_list:
            x1 = self.layer5_list[i](x, get_feat=True)  # produce segmap
            outputs.append(x1)

        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_1x_lr_params(self):

        b = []
        for layer in self.features:
            b.append(layer.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_1x_lr_params_new(self):

        b = []
        for layer in self.features:
            b.append(layer.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

        b1 = []
        if self.bn_clr:
            b1.append(self.bn_pretrain.parameters())

        for layer in self.layer5_list:
            if layer != self.layer5_list[-2]:  ## -1
                b1.append(layer.parameters())

        for j in range(len(b1)):
            for i in b1[j]:
                yield i

    def get_10x_lr_params(self):

        b = []
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())

            # b.append(self.layer5.parameters())
        for layer in self.layer5_list:
            b.append(layer.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_10x_lr_params_new(self):

        b = []
        layer = self.layer5_list[-2]  ##-1
        b.append(layer.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters_new(self, args):
        return [{'params': self.get_1x_lr_params_new(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params_new(), 'lr': 10 * args.learning_rate}]

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss


def DeeplabVGG(BatchNorm, num_classes=7, num_target=1, freeze_bn=False, restore_from=None, initialization=None,
               bn_clr=False, stage=None):
    vgg = models.vgg16()
    model = VGG(vgg, num_classes, BatchNorm, num_target=num_target, bn_clr=bn_clr, stage=stage)

    if freeze_bn:
        model.apply(freeze_bn_func)
    if initialization is None:
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
    else:
        pretrain_dict = torch.load(initialization)['state_dict']
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)

    if restore_from is not None:
        checkpoint = torch.load(restore_from)['VGG']["model_state"]
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    return model


class SegFormer(nn.Module):
    def __init__(self, num_classes, BatchNorm, num_target=1, bn_clr=False, stage=None, phi='b5', pretrained=True):
        super(SegFormer, self).__init__()

        self.bn_clr = bn_clr
        self.num_target = num_target
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = get_mit_backbone(phi, pretrained)

        self.channels = 256
        self.in_index = [0, 1, 2, 3]
        self.dropout_ratio = 0.1
        self.num_classes = num_classes
        self.align_corners = False

        self.layer5_list = nn.ModuleList()
        if stage == 'stage1':
            for i in range(self.num_target):
                if i != self.num_target - 1:
                    layer = self._make_pred_layer(Classifier_Module2, 1024, [6, 12, 18, 24], [6, 12, 18, 24],
                                                  self.num_classes)
                else:
                    layer = self._make_pred_layer(Classifier_Module2, 1024 * (self.num_target - 2), [6, 12, 18, 24],
                                                  [6, 12, 18, 24], self.num_classes)  ## ensemble
                self.layer5_list.append(layer)
        else:
            for i in range(self.num_target):
                layer = self._make_pred_layer(Classifier_Module2, 1024, [6, 12, 18, 24], [6, 12, 18, 24], self.num_classes)
                self.layer5_list.append(layer)

        if self.bn_clr:
            self.bn_pretrain = BatchNorm(1024, affine=affine_par)

        for m in self.layer5_list.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, domain_list, ssl, target_ensembel=False, ensembel=False):

        if ensembel:
            if target_ensembel:
                x = torch.concat(ssl, dim=1)  # torch.concat(torch.split(x, b//(self.num_target-2), 0), dim=1)
            else:
                x = torch.concat(ssl[:-1], dim=1)  ## the last feature is domain-agnostic
        else:
            x = self.backbone.forward(x)
            inputs = [x[i] for i in self.in_index]
            upsampled_inputs = [resize(input=x, size=inputs[0].shape[2:],
                                       mode='bilinear', align_corners=self.align_corners) for x in inputs]
            x = torch.cat(upsampled_inputs, dim=1)

            ssl.append(x)
            if self.bn_clr:
                x = self.bn_pretrain(x)

        outputs = []
        for i in domain_list:
            x1 = self.layer5_list[i](x, get_feat=True)  # produce segmap
            outputs.append(x1)

        return outputs

    def get_1x_lr_params(self):

        b = []
        b.append(self.backbone)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())
        for layer in self.layer5_list:
            b.append(layer.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_1x_lr_params_new(self):

        b = []
        b.append(self.backbone)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

        b1 = []
        if self.bn_clr:
            b1.append(self.bn_pretrain.parameters())

        for layer in self.layer5_list:
            if layer != self.layer5_list[-2]:  ## -1
                b1.append(layer.parameters())

        for j in range(len(b1)):
            for i in b1[j]:
                yield i

    def get_10x_lr_params_new(self):

        b = []
        layer = self.layer5_list[-2]
        b.append(layer.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters_new(self, args):
        return [{'params': self.get_1x_lr_params_new(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params_new(), 'lr': 10 * args.learning_rate}]

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss


def DeeplabSegFormer(BatchNorm, num_classes=7, num_target=1, freeze_bn=False, restore_from=None, initialization=None,
                     bn_clr=False, stage=None):
    model = SegFormer(num_classes, BatchNorm, num_target=num_target, bn_clr=bn_clr, stage=stage, phi='b5')

    if freeze_bn:
        model.apply(freeze_bn_func)

    if restore_from is not None:
        checkpoint = torch.load(restore_from)['SegFormer']["model_state"]
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    return model



