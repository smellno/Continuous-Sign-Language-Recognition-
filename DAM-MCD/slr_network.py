import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss


import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
# import modules.resnet2 as Resnet_moudle
import modules.resnet2 as resnet
from modules.s3d import s3de
from f import attention_fusion

# Resnet = Resnet_moudle.Resnet()
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.classifier1 = nn.Linear(512, 1086)

    def forward(self, x):
        # x_logits  = self.classifier1(x.view(batch, temp, -1).permute(1,0,2))
        return x

        #     "visual_feat": x.view(batch, temp, -1).permute(0,2,1),
        #     "conv_logits": x_logits,
        #     "feat_len": lgt.cpu(),
        # }


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        # self.conv2d = getattr(resnet, c2d_type)()

        self.conv2d = getattr(resnet, c2d_type)()
        # self.conv2d = Resnet_moudle.Resnet()
        self.conv2d.fc = Identity()
        # self.atten = attention_fusion()

        self.s3d = s3de(self.num_classes)

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.conv1d2 = TemporalConv(input_size=1024,
                                   hidden_size=hidden_size,
                                   conv_type=1,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.classifier1 = nn.Linear(512, self.num_classes)
        self.classifier2 = nn.Linear(1024, self.num_classes)

        # self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    #     def masked_bn(self, inputs, len_x):
    #         def pad(tensor, length):
    #             return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    #         x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
    #         print('x1',x.shape)
    #         print('x1',x)

    #         x = self.conv2d(x)
    #         print('x2',x.size)
    #         print('x2',x)

    #         x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
    #                        for idx, lgt in enumerate(len_x)])
    #         print('x3',x)
    #         print('x3',x.size)

    #         return x

    #     def masked_bn(self, inputs, len_x):
    #             def pad(tensor, length):
    #                 return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    #             x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
    #             print('x1',x.shape)
    #             print('x1',x)
    #             x = self.conv2d(x)
    #             print('x2',x.shape)
    #             print('x2',x)
    #             x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
    #                            for idx, lgt in enumerate(len_x)])
    #             print('x3',x.shape)
    #             print('x3',x)
    #             return x

    def forward(self, x, len_x, x1, len_x1, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            # inputs = x.reshape(batch * temp, channel, height, width)
            # print('inputs',inputs)
            # print('inputs.shape',inputs.shape)

            print('len_x', len_x)
            print('len_x1', len_x1)

            # framewise = self.masked_bn(inputs, len_x)
            # print('framewise',framewise)
            # print('framewise-shape',framewise.shape)

            s3d_feature = self.s3d(x1.permute(0, 2, 1, 3, 4))

            # framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4)).view(batch, temp, -1)  # btc -> bct
            print('framewise-shape', framewise.shape)

            framewise1 = framewise.permute(0, 2, 1)
            framewise2 = framewise.permute(1, 0, 2)
            print('framewise2-shape', framewise2.shape)

            framewise2 = self.classifier1(framewise2)
            print('framewise2-shape', framewise2.shape)

            # framewise = torch.from_numpy(framewise)
            # framewise = framewise.to(torch.float16)  # 将输入数据转换为 float16（半精度浮点数）
        else:
            # frame-wise features
            framewise = x
            framewise = torch.from_numpy(framewise)
            print('framewise', framewise)

        if framewise.size(0) == 0:
            # 如果输入为空，可以选择跳过 conv1d 的处理
            print("Input is empty. Skipping conv1d.")
        else:
            # 否则，继续执行 conv1d
            print('framewise1-shape', framewise1.shape)
            conv1d_outputs = self.conv1d(framewise1, len_x)
            x = conv1d_outputs['visual_feat']
            x = x.permute(1, 2, 0)
            len_x = conv1d_outputs['feat_len1']

            conv1d_outputs2 = self.conv1d2(x, len_x)
            # x: T, B, C
            x = conv1d_outputs2['visual_feat']
            x3d = s3d_feature['conv_logits']
            x3d = x3d.permute(2, 0, 1)
            print('x-shape', x.shape)
            print('x3d-shape', x3d.shape)
            result = attention_fusion(x, x3d)
            result1 = self.classifier2(result)

            lgt = conv1d_outputs2['feat_len']
            tm_outputs = self.temporal_model(result, lgt)
            print('tm_outputs.predictions', tm_outputs['predictions'])
            print('tm_outputs.predictions', tm_outputs['predictions'].size())

            print('tm_outputs.hidden', tm_outputs['hidden'])
            print('tm_outputs.hidden', tm_outputs['hidden'].size())

            outputs = self.classifier(tm_outputs['predictions'])
            print(outputs)
            print('outputs.size', outputs.size())

            conv1d_output11s = conv1d_outputs2['conv_logits']
            print('conv1d_output11s.shape', conv1d_output11s.shape)
            print('s3d_feature[logits].permute(2,0,1).shape', s3d_feature['logits'].permute(2, 0, 1).shape)
            print('s3d_feature[lgt]', s3d_feature['lgt'])
            print('[lgt]', lgt)

            pred = None if self.training \
                else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
            conv_pred = None if self.training \
                else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

            print('pred', pred)
            print('conv_pred', conv_pred)

            return {
                "framewise_features": framewise2,
                "visual_features3d+2d": result,
                "feat_len3d": conv1d_outputs2['feat_len'],
                "feat_len": lgt,
                "conv_logits": conv1d_outputs2['conv_logits'],
                "sequence_logits": outputs,
                "conv_sents": conv_pred,
                "recognized_sents": pred,
            }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        if ret_dict is not None and 'sequence_logits' in ret_dict and ret_dict["sequence_logits"] is not None:
            for k, weight in self.loss_weights.items():
                if k == 'ConvCTC':
                    loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                          label.cpu().int(), ret_dict["feat_len3d"].cpu().int(),
                                                          label_lgt.cpu().int()).mean()
                    loss += weight * self.loss['CTCLoss'](ret_dict["visual_features3d+2d"].log_softmax(-1),
                                                          label.cpu().int(), ret_dict["feat_len3d"].cpu().int(),
                                                          label_lgt.cpu().int()).mean()
                    loss += weight * self.loss['CTCLoss'](ret_dict["framewise_features"].log_softmax(-1),
                                                          label.cpu().int(), ret_dict["feat_len3d"].cpu().int(),
                                                          label_lgt.cpu().int()).mean()
                elif k == 'SeqCTC':
                    loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                          label.cpu().int(), ret_dict["feat_len3d"].cpu().int(),
                                                          label_lgt.cpu().int()).mean()
                    loss += weight * self.loss['CTCLoss'](ret_dict["visual_features3d+2d"].log_softmax(-1),
                                                          label.cpu().int(), ret_dict["feat_len3d"].cpu().int(),
                                                          label_lgt.cpu().int()).mean()
                elif k == 'Dist':
                    loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                               ret_dict["sequence_logits"].detach(),
                                                               use_blank=False)
        else:
            print('CTCLoss is not defined in self.loss')
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
