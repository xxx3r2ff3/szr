# -*- coding: utf-8 -*-
# perceptual —— S004 恢复(landmark2face_wy/loss/perceptual.pyc,py3.8;2031 行反汇编,
# 14 个 code object)。依据:
# evidence/modules/_heygem_orphans/disasm/landmark2face_wy_loss_perceptual.pyc.dis
#
# 逐条按 [Disassembly] 重建的要点:
#   - apply_imagenet_normalization: (input + 1) / 2 后再减均值除方差(注意 +1/2 这一步)。
#   - PerceptualLoss.__init__ 的 11 参签名与默认值元组
#     ('vgg19', 'relu_4_1', None, 'l1', False, 'bilinear', False, 1, False, 0)
#     (默认值元组只有 10 项,少的一项为 self;weight_style_to_perceptual 默认 0)。
#   - PerceptualLoss.forward 的 mask 形状用 input_feature.shape[2:] 取,
#     style_loss 仅在 use_style_loss and scale == 0 时累加;
#     scale != num_scales - 1 时以 0.5 为 scale_factor 继续降采样。
#     末尾 `return None` 对应 LOAD_CONST None; RETURN_VALUE,为原文件中的不可达收尾。
#   - _vgg19/_vgg16/_alexnet/_inception_v3/_resnet50/_robust_resnet50/_vgg_face_dag
#     的层名映射表与权重 URL 全部逐字取自各自 code object 的 [Constants]。
#   - _vgg_face_dag 内部定义局部类 Flatten(nn.Module),其 __qualname__
#     '_vgg_face_dag.<locals>.Flatten' 由嵌套定义自然产生。
# 未还原细节:无。仅 _vgg_face_dag 中 nn.Sequential(*seq_layers) 使用
#   CALL_FUNCTION_EX(0) 的星号展开形式,已按语义写成 *seq_layers。

import torch
import torch.nn.functional
from torch.nn import functional as F
import torchvision
from torch import nn


def apply_imagenet_normalization(input):
    """Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    normalized_input = (input + 1) / 2
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


class PerceptualLoss(nn.Module):
    """Perceptual loss initialization.

    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the input images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None, criterion='l1',
                 resize=False, resize_mode='bilinear', instance_normalized=False, num_scales=1,
                 use_style_loss=False, weight_style_to_perceptual=0):
        super(PerceptualLoss, self).__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]
        if not len(layers) == len(weights):
            raise AssertionError('The number of layers (%s) must be equal to the number of weights (%s).'
                                 % (len(layers), len(weights)))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        elif network == 'alexnet':
            self.model = _alexnet(layers)
        elif network == 'inception_v3':
            self.model = _inception_v3(layers)
        elif network == 'resnet50':
            self.model = _resnet50(layers)
        elif network == 'robust_resnet50':
            self.model = _robust_resnet50(layers)
        elif network == 'vgg_face_dag':
            self.model = _vgg_face_dag(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)
        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        self.use_style_loss = use_style_loss
        self.weight_style = weight_style_to_perceptual
        print('Perceptual loss:')
        print('\tMode: {}'.format(network))

    def forward(self, inp, target, mask=None):
        """Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        self.model.eval()
        inp = apply_imagenet_normalization(inp)
        target = apply_imagenet_normalization(target)
        if self.resize:
            inp = F.interpolate(inp, mode=self.resize_mode, size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode=self.resize_mode, size=(224, 224), align_corners=False)
        loss = 0
        style_loss = 0
        for scale in range(self.num_scales):
            input_features = self.model(inp)
            target_features = self.model(target)
            for layer, weight in zip(self.layers, self.weights):
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)
                if mask is not None:
                    mask_ = F.interpolate(mask, mode='bilinear',
                                          size=input_feature.shape[2:], align_corners=False)
                    input_feature = input_feature * mask_
                    target_feature = target_feature * mask_
                loss += weight * self.criterion(input_feature, target_feature)
                if self.use_style_loss and scale == 0:
                    style_loss += self.criterion(self.compute_gram(input_feature),
                                                 self.compute_gram(target_feature))
            if scale != self.num_scales - 1:
                inp = F.interpolate(inp, mode=self.resize_mode, scale_factor=0.5,
                                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(target, mode=self.resize_mode, scale_factor=0.5,
                                       align_corners=False, recompute_scale_factor=True)
        if self.use_style_loss:
            return loss + style_loss * self.weight_style
        return loss

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G


class _PerceptualNetwork(nn.Module):
    """The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super(_PerceptualNetwork, self).__init__()
        if not isinstance(network, nn.Sequential):
            raise AssertionError('The network needs to be of type "nn.Sequential".')
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


def _vgg19(layers):
    """Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg16(layers):
    """Get vgg16 layers"""
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          18: 'relu_4_1',
                          20: 'relu_4_2',
                          22: 'relu_4_3',
                          25: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _alexnet(layers):
    """Get alexnet layers"""
    network = torchvision.models.alexnet(pretrained=True).features
    layer_name_mapping = {0: 'conv_1',
                          1: 'relu_1',
                          3: 'conv_2',
                          4: 'relu_2',
                          6: 'conv_3',
                          7: 'relu_3',
                          8: 'conv_4',
                          9: 'relu_4',
                          10: 'conv_5',
                          11: 'relu_5'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _inception_v3(layers):
    """Get inception v3 layers"""
    inception = torchvision.models.inception_v3(pretrained=True)
    network = nn.Sequential(inception.Conv2d_1a_3x3,
                            inception.Conv2d_2a_3x3,
                            inception.Conv2d_2b_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Conv2d_3b_1x1,
                            inception.Conv2d_4a_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Mixed_5b,
                            inception.Mixed_5c,
                            inception.Mixed_5d,
                            inception.Mixed_6a,
                            inception.Mixed_6b,
                            inception.Mixed_6c,
                            inception.Mixed_6d,
                            inception.Mixed_6e,
                            inception.Mixed_7a,
                            inception.Mixed_7b,
                            inception.Mixed_7c,
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    layer_name_mapping = {3: 'pool_1',
                          6: 'pool_2',
                          14: 'mixed_6e',
                          18: 'pool_3'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _resnet50(layers):
    """Get resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=True)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _robust_resnet50(layers):
    """Get robust resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.utils.model_zoo.load_url('http://andrewilyas.com/ImageNet.pt')
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith('module.model.'):
            new_state_dict[k[13:]] = v
    resnet50.load_state_dict(new_state_dict)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face_dag(layers):
    """Get vgg face layers"""
    network = torchvision.models.vgg16(num_classes=2622)
    state_dict = torch.utils.model_zoo.load_url(
        'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth')
    feature_layer_name_mapping = {0: 'conv1_1',
                                  2: 'conv1_2',
                                  5: 'conv2_1',
                                  7: 'conv2_2',
                                  10: 'conv3_1',
                                  12: 'conv3_2',
                                  14: 'conv3_3',
                                  17: 'conv4_1',
                                  19: 'conv4_2',
                                  21: 'conv4_3',
                                  24: 'conv5_1',
                                  26: 'conv5_2',
                                  28: 'conv5_3'}
    new_state_dict = {}
    for k, v in feature_layer_name_mapping.items():
        new_state_dict['features.' + str(k) + '.weight'] = state_dict[v + '.weight']
        new_state_dict['features.' + str(k) + '.bias'] = state_dict[v + '.bias']

    classifier_layer_name_mapping = {0: 'fc6',
                                     3: 'fc7',
                                     6: 'fc8'}
    for k, v in classifier_layer_name_mapping.items():
        new_state_dict['classifier.' + str(k) + '.weight'] = state_dict[v + '.weight']
        new_state_dict['classifier.' + str(k) + '.bias'] = state_dict[v + '.bias']

    network.load_state_dict(new_state_dict)

    class Flatten(nn.Module):
        """Flatten the tensor"""

        def forward(self, x):
            """Flatten it"""
            return x.view(x.shape[0], -1)

    layer_name_mapping = {1: 'avgpool',
                          3: 'fc6',
                          4: 'relu_6',
                          6: 'fc7',
                          7: 'relu_7',
                          9: 'fc8'}
    seq_layers = [network.features, network.avgpool, Flatten()]
    for i in range(7):
        seq_layers += [network.classifier[i]]
    network = nn.Sequential(*seq_layers)
    return _PerceptualNetwork(network, layer_name_mapping, layers)
