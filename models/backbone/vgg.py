import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, use_decoder=False):
        super(VGG, self).__init__()
        self.features_dict = features

        self.decoder = use_decoder
        self.features_1 = self.features_dict['features_1']
        self.features_2 = self.features_dict['features_2']
        self.features_3 = self.features_dict['features_3']
        self.features_4 = self.features_dict['features_4']
        self.features_5 = self.features_dict['features_5']

        self.cls = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # fc6
        )

        if self.decoder:
            self.side3_1 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, dilation=1, padding=1),  # fc6
                nn.ReLU()
            )
            self.side4_1 = nn.Sequential(
                nn.Conv2d(640, 128, kernel_size=3, dilation=1, padding=1),  # fc6
                nn.ReLU(),
            )
            self.side5_1 = nn.Sequential(
                nn.Conv2d(640, 128, kernel_size=3, dilation=1, padding=1),  # fc6
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.features_1(x)
        x = self.features_2(x)
        out3 = self.features_3(x)
        out4 = self.features_4(out3)
        out5 = self.features_5(out4)

        out = self.cls(out5)

        if self.decoder:
            side_out3_1 = self.side3_1(out3)
            side_out4_1 = self.side4_1(torch.cat((side_out3_1, out4), dim=1))
            side_out5_1 = self.side5_1(torch.cat((side_out4_1, out5), dim=1))

            return out, side_out5_1
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, dilation=None, batch_norm=False, in_channels=3):
    layers = []
    # in_channels = 3
    # in_channels = 3
    layer_dict = OrderedDict()
    layer_count = 0
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            layer_count += 1
            layer_dict['features_%d' % (layer_count)] = nn.Sequential(*layers)
            layers = []

        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            layer_count += 1
            layer_dict['features_%d' % (layer_count)] = nn.Sequential(*layers)
            layers = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # return nn.Sequential(*layers)

    return layer_dict


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_deeplab': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
}

dilation = {
    'D': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N'],
    'E': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 1, 'M', 1, 1, 1, 1, 'N', 1, 1, 1, 1, 'N']  # 'M'\'M'
}


def restore(model, pre_model):
    try:
        model.load_state_dict(pre_model)
    except RuntimeError:
        print("KeyError")
        model_dict = model.state_dict()
        new_model_dict = OrderedDict()
        for k_model, k_pretrained in zip(list(model_dict.keys()),
                                         list(pre_model.keys())[:len(list(model_dict.keys()))]):
            if model_dict[k_model].size() == pre_model[k_pretrained].size():
                print("%s\t<--->\t%s" % (k_model, k_pretrained))
                new_model_dict[k_model] = pre_model[k_pretrained]
            else:
                print('Fail to load %s' % (k_model))

        model_dict.update(new_model_dict)
        model.load_state_dict(model_dict)
    except KeyError:
        print("Loading pre-trained values failed.")
        raise


# else:
#     print("=> no checkpoint found at '{}'".format(snapshot))

def vgg16(pretrained=False, in_channels=3, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['D_deeplab'], dilation=dilation['D'], in_channels=in_channels),
                **kwargs)  # origin #D_deeplab
    # model = VGG(make_layers(cfg['E'], dilation=dilation['E'], in_channels=in_channels), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        restore(model, model_zoo.load_url(model_urls['vgg16']))  # origin
        # restore(model, model_zoo.load_url(model_urls['vgg19']))

    return model


def vgg19(pretrained=False, in_channels=3, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    # model = VGG(make_layers(cfg['D_deeplab'], dilation=dilation['D'], in_channels=in_channels), **kwargs) # origin
    model = VGG(make_layers(cfg['E'], dilation=dilation['E'], in_channels=in_channels), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        # restore(model, model_zoo.load_url(model_urls['vgg16'])) #origin
        restore(model, model_zoo.load_url(model_urls['vgg19']))

    return model


