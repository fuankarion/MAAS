import torch
import torch.nn as nn
import torch.nn.parameter
from torch import Tensor
from typing import Callable, Optional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock2D(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck2D(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class TwoStreamResNet(nn.Module):
    def __init__(self, block, layers, rgb_stack_size, num_classes=2,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(TwoStreamResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Audio stream
        self.inplanes = 64
        self.audio_conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2,
                                     padding=3, bias=False)
        self.a_bn1 = norm_layer(self.inplanes)
        self.a_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a_layer1 = self._make_layer(block, 64, layers[0])
        self.a_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.a_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.a_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Video stream
        self.inplanes = 64
        self.video_conv1 = nn.Conv2d(3*rgb_stack_size, self.inplanes,
                                     kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.v_bn1 = norm_layer(self.inplanes)
        self.v_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.v_layer1 = self._make_layer(block, 64, layers[0])
        self.v_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.v_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.v_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Shared
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_128_a = nn.Linear(512 * block.expansion, 128)
        self.fc_128_v = nn.Linear(512 * block.expansion, 128)

        # Predictions
        self.fc_final = nn.Linear(128*2, num_classes)
        self.fc_aux_a = nn.Linear(128, num_classes)
        self.fc_aux_v = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock2D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, a, v):
        # Audio Stream
        a = self.audio_conv1(a)
        a = self.a_bn1(a)
        a = self.relu(a)
        a = self.maxpool(a)

        a = self.a_layer1(a)
        a = self.a_layer2(a)
        a = self.a_layer3(a)
        a = self.a_layer4(a)
        a = self.avgpool(a)

        # Video Stream
        v = self.video_conv1(v)
        v = self.v_bn1(v)
        v = self.relu(v)
        v = self.maxpool(v)

        v = self.v_layer1(v)
        v = self.v_layer2(v)
        v = self.v_layer3(v)
        v = self.v_layer4(v)
        v = self.avgpool(v)

        # Concat Stream Feats
        a = a.reshape(a.size(0), -1)
        v = v.reshape(v.size(0), -1)
        stream_feats = torch.cat((a, v), 1)

        # Auxiliary supervisions
        a = self.fc_128_a(a)
        a = self.relu(a)
        v = self.fc_128_v(v)
        v = self.relu(v)

        aux_a = self.fc_aux_a(a)
        aux_v = self.fc_aux_v(v)

        # Global supervision
        av = torch.cat((a, v), 1)
        x = self.fc_final(av)

        return x, aux_a, aux_v, stream_feats[..., :512], stream_feats[..., 512:]


# Utility
def _load_weights_into_two_stream_resnet(model, rgb_stack_size, arch, progress):
    resnet_state_dict = load_state_dict_from_url(model_urls[arch],
                                                 progress=progress)

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():

        # this cases are not mutually exlcusive
        if 'v_'+name in own_state:
            own_state['v_'+name].copy_(param)
        if 'a_'+name in own_state:
            own_state['a_'+name].copy_(param)
        if 'v_'+name not in own_state and 'a_'+name not in own_state:
            print('No assignation for ', name)

    conv1_weights = resnet_state_dict['conv1.weight']
    tupleWs = tuple(conv1_weights for i in range(rgb_stack_size))
    concatWs = torch.cat(tupleWs, dim=1)
    own_state['video_conv1.weight'].copy_(concatWs)

    avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
    own_state['audio_conv1.weight'].copy_(avgWs)

    print('loaded ws from resnet')
    return model


# Resnet any size
def _two_stream_resnet(arch, block, layers, pretrained, progress, rgb_stack_size,
                       num_classes, **kwargs):
    model = TwoStreamResNet(block, layers, rgb_stack_size, num_classes, **kwargs)

    if pretrained:
        model = _load_weights_into_two_stream_resnet(model, rgb_stack_size,
                                                     arch, progress)
    return model


# Prefer these callable methods
def resnet18_two_streams(pretrained=False, progress=True, rgb_stack_size=5,
                         num_classes=2, **kwargs):
    return _two_stream_resnet('resnet18', BasicBlock2D, [2, 2, 2, 2], pretrained,
                              progress, rgb_stack_size, num_classes, **kwargs)


def resnet18_two_streams_forward(pretrained_weights_path, progress=True,
                                 rgb_stack_size=5, num_classes=2, **kwargs):
    model = _two_stream_resnet('resnet18', BasicBlock2D, [2, 2, 2, 2], False, progress,
                               rgb_stack_size, num_classes, **kwargs)
    model.load_state_dict(torch.load(pretrained_weights_path))
    model.eval()
    return model


def resnet34_two_streams(pretrained=False, progress=True, rgb_stack_size=5,
                         num_classes=2, **kwargs):
    return _two_stream_resnet('resnet34', BasicBloc2Dk, [3, 4, 6, 3], pretrained,
                              progress, rgb_stack_size, num_classes, **kwargs)


def resnet50_two_streams(pretrained=False, progress=True, rgb_stack_size=5,
                         num_classes=2, **kwargs):
    return _two_stream_resnet('resnet50', Bottleneck2D, [3, 4, 6, 3], pretrained,
                              progress, rgb_stack_size, num_classes, **kwargs)
