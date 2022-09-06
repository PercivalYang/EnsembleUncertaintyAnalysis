from torchensemble.utils import io
from models.r18 import ResNet18
from torchensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import os
import torch
import matplotlib.pyplot as plt

from utils import ForwardStdEntropyEvalute


def EvaluateUncertainty(model, DataTotal):
    valid_loader = DataTotal['Test']
    Labels = []
    Std = []
    Entropy = []
    PredRight = []
    for data, label in tqdm(valid_loader):
        data, label = data.cuda(), label.cuda()
        metrics, pred = ForwardStdEntropyEvalute(model, data)

        Std.append(metrics['std'])
        Entropy.append(metrics['entropy'])
        Labels.append(label)
        pred = pred.mean(-1)    # (batch_size, num_class, num_estimators), 对num_estimators求平均即集成网络输出
        PredRight.append((pred.argmax(1) == label))

    print('done')


def EvaluateROC(model, DataTotal, args):
    ModeName = ['Train', 'Test']

    for ModeNow in ModeName:
        Labels = []
        PredsStra = []
        data_loader = DataTotal[ModeNow]

        for data, label in tqdm(data_loader):
            data, label = data.cuda(), label.cuda()
            Preds = model.predict(data)  # 模型输出预测结果
            PredsStra.append(Preds[:, 1])  # 将每个Batch的结果保存到list中
            Labels.append(label)

        PredsStra = torch.cat(PredsStra).cpu()
        Labels = torch.cat(Labels).cpu()  # 将list变换为tensor

        FPR, TPR, Thresholds = roc_curve(Labels, PredsStra, pos_label=1)
        ROC_AUC = auc(FPR, TPR)

        plt.plot(FPR, TPR, label='{} Area={:.2f}'.format(ModeNow, ROC_AUC))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("./EvaluateResults/ROCfig/{}_{}.jpg".format(args.net,args.n_estimators))

def EvaluateACC(model, TestData):
    """Docstrings decorated by downstream models."""
    model.eval()
    correct = 0
    total = 0

    for data, target in tqdm(TestData):
        data, target = data.cuda(),target
        output = model.predict(data)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    acc = 100 * correct / total

    return acc

def AUCWithNetNums(model, Data):
    pass


def AUCWithTTAUGNums(model, Data):
    pass
