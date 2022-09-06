import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import PIL.Image as Image

import torch
import os

from torch.utils.data import DataLoader
from torchvision import datasets

class AverageMeter(object):
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


def ImageTransforms(mode):
    if mode == 'full':
        image_transforms = {
            'Train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'Test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }
        return image_transforms
    elif mode == 'cropped':
        image_transforms = {
            'Train': transforms.Compose([
                Expand2SquareImg(), # 其中自带Resize((224,224))
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'Test': transforms.Compose([
                Expand2SquareImg(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }
        return image_transforms

def DataLoad(DatasetPath, batch_size, img_mode):
    '''

    :param DatasetPath: Full eyes(~/Datasets/TrainValidDataset), Cropped eyes(~/Datasets/Cropped)
    :param batch_size:
    :return:
    '''
    Train_directory = os.path.join(DatasetPath, 'train')
    test_directory = os.path.join(DatasetPath, 'valid')

    data = {
        'Train': datasets.ImageFolder(root=Train_directory, transform=ImageTransforms(img_mode)['Train']),
        'Test': datasets.ImageFolder(root=test_directory, transform=ImageTransforms(img_mode)['Test'])
    }

    Train_data = DataLoader(data['Train'], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_data = DataLoader(data['Test'], batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    DataTotal = {"Train":Train_data,"Test":test_data}

    return DataTotal

def AddDropoutLayer(Net):
    feats_list = list(Net.features)
    new_feats_list = []
    for feat in feats_list:
        new_feats_list.append(feat)
        if isinstance(feat, nn.Conv2d):
            new_feats_list.append(nn.Dropout(p=0.5, inplace=True))
    Net.features = nn.Sequential(*new_feats_list)
    return Net


def ForwardStdEntropyEvalute(model, data):

    outputs_list = []
    for estimator in model.estimators_:
        estimator.eval()
        with torch.no_grad():
            outputs_list.append(F.softmax(estimator(data), dim=1))
            
    Outputs = torch.stack(outputs_list,2)
    LogOut = Outputs.clamp_min(1e-6).log()
    OutStd = Outputs.std(dim=2).mean(dim=1)
    OutEntropy = Outputs * LogOut

    Metrics = {'std': OutStd,'entropy':OutEntropy}
    return Metrics, Outputs

class Expand2SquareImg:
    def __call__(self, PILImg):
        width, height = PILImg.size
        if width == height:
            return PILImg
        elif width > height:
            new_img = Image.new(PILImg.mode, (width,width)) # 创建一个正方形黑色背景
            new_img.paste(PILImg, (0, (width - height) // 2))   # 将原图粘贴过来
        else:
            new_img = Image.new(PILImg.mode, (width,width))
            new_img.paste(PILImg, (0, (height - width) // 2, 0))
        new_img = new_img.resize((224,224))
        return new_img