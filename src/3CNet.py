from glob import glob
import os
import codecs
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm_notebook as tqdm
from fastai.vision import *
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from fastai.vision import create_head
from torchvision.models.resnet import ResNet, Bottleneck
from sklearn.model_selection import StratifiedKFold
from fastai.callbacks import *
from torchvision import transforms
from fastai.vision import Image
import torchvision
from PIL import Image
import numpy as np
from thop import profile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

from fastai.tabular import *
from torch.utils.data.sampler import WeightedRandomSampler


class OverSamplingCallback(LearnerCallback):
    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.labels = self.learn.data.train_dl.dataset.y.items
        _, counts = np.unique(self.labels, return_counts=True)
        self.weights = torch.DoubleTensor((1 / counts)[self.labels])
        self.label_counts = np.bincount([self.learn.data.train_dl.dataset.y[i].data
                                         for i in range(len(self.learn.data.train_dl.dataset))])
        self.total_len_oversample = int(self.learn.data.c * np.max(self.label_counts))

    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(
            WeightedRandomSampler(weights, self.total_len_oversample),
            self.learn.data.train_dl.batch_size, False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
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

    def forward(self, x):
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class ClusterModel(nn.Module):
    def __init__(self, arch, concat_pool, dropout, bn_final):
        super(ClusterModel, self).__init__()
        self.body = nn.Sequential(*(list(arch.children())[:-2]))
        self.head = create_head(4096, 4, concat_pool=concat_pool == 1, ps=dropout, bn_final=bn_final == 1)
        # print(self.head)

    def forward(self, x):
        x = self.body(x)
        return self.head(x)


class Specificity(Callback):

    def __init__(self):
        super().__init__()
        self.name = "specificity"

    def on_epoch_begin(self, **kwargs):
        self.cm = None
        self.n_classes = 4

    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = last_output.argmax(-1).view(-1).cpu()
        targs = last_target.cpu()
        self.n_classes = last_output.shape[-1]
        self.x = torch.arange(0, self.n_classes)
        cm = ((preds == self.x[:, None]) & (targs == self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None:
            self.cm = cm
        else:
            self.cm += cm

    def on_epoch_end(self, last_metrics, **kwargs):
        self.tp, self.tn, self.fn, self.fp = 0.0, 0.0, 0.0, 0.0
        elems = self.cm.shape[0]
        for elem in range(0, elems):
            self.tp += self.cm[elem, elem]
            for x in range(0, elems):
                for y in range(0, elems):
                    if x == elem and y != elem:
                        self.fn += self.cm[x, y]
                    if x != elem and y == elem:
                        self.fp += self.cm[x, y]
                    if x != elem and y != elem:
                        self.tn += self.cm[x, y]
        specificity = self.tn / (self.tn + self.fp)
        return add_metrics(last_metrics, specificity)


def _get_model(state_dict, block, layers, concat_pool, dropout, bn_final, **kwargs):
    model = ResNet(block, layers, **kwargs)
    model.load_state_dict(state_dict)
    model = ClusterModel(model, concat_pool, dropout, bn_final)
    return model


def get_cluster_model(state_dict, concat_pool, dropout, bn_final, width=16, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = width
    return _get_model(state_dict, Bottleneck, [3, 4, 23, 3], concat_pool, dropout, bn_final, **kwargs)


if __name__ == '__main__':
    img_paths = []
    labels = []
    img_path = 'resize'
    for dir_path, dir_names, file_names in os.walk(img_path):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            img_paths.append(file_path)
            labels.append(int(file_path.split('/')[1]))
    df = pd.DataFrame({'name': img_paths, 'label': labels})
    df = df.reindex(['name', 'label'], axis=1)

    for train_index, test_index in StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2,
                                                          random_state=42).split(
            df, df['label']):
        testing_df = df.iloc[test_index]
        train_df = df.iloc[train_index]
        val_df, test_df = train_test_split(testing_df, test_size=0.5, stratify=testing_df['label'], random_state=42)
        print(train_df.shape, val_df.shape, test_df.shape)

    full_df = train_df.append(val_df)
    full_df['is_valid'] = 0
    full_df.loc[val_df.index, 'is_valid'] = 1
    full_df = full_df.reset_index(drop=True)
    full_df['is_valid'].value_counts()
    data = (ImageList
            .from_df(full_df, '.')
            .split_by_idx(valid_idx=full_df[full_df['is_valid'] == 1].index)
            .label_from_df(cols='label')
            .transform(get_transforms(), size=224)
            .databunch(bs=16, num_workers=8)
            .normalize(imagenet_stats)
            )

    data_test = (ImageList
                 .from_df(df, '.')
                 .split_by_idx(valid_idx=test_df.index)
                 .label_from_df(cols='label')
                 .transform(get_transforms(), size=224)
                 .databunch(bs=16, num_workers=8)
                 .normalize(imagenet_stats)
                 )
    loss = LabelSmoothingCrossEntropy(0.1)
    my_fbeta = FBeta(average='macro')
    recall = Recall(average='macro')
    precision = Precision(average='macro')
    kappa_score = KappaScore()
    auroc = AUROC()
    max_epochs = 10
    max_lr = 1e-5
    min_delta = 0.01
    patience = 5

    model = get_cluster_model(torch.load('ig_resnext101_32x8-c38310e5.pth'),
                                       width=8, concat_pool=1, dropout=0.5, bn_final=0)
    learn = Learner(data, model,
                    metrics=[precision, accuracy, my_fbeta, kappa_score, recall, Specificity(), auroc],
                    loss_func=loss,
                    callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=min_delta, patience=patience)],
                    model_dir='./',
                    path='./')
    learn = learn.split(lambda m: (m.body[6], m.head))
    learn.freeze_to(-1)
    learn.fit_one_cycle(5, max_lr=slice(1e-5, 4e-5),, callbacks=[
        SaveModelCallback(learn, every='improvement', monitor='accuracy', name='proposed_101_32x8d-best')])
    learn.unfreeze()
    learn.fit_one_cycle(10, max_lr=slice(1e-6, 4e-6), 
                    callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name='proposed_101_32x8d-best')])
    learn.restore_best_weights = True
    result = learn.validate(data_test.valid_dl)
    print('precision_result:', result[1].item())
    print('accuracy_result:', result[2].item())
    print('f_beta:', result[3].item())
    print('kappa_score_result:', result[4].item())
    print('recall_result:', result[5].item())
    print('specificity_result:', result[6].item())
    print('auroc_result:', result[7].item())

