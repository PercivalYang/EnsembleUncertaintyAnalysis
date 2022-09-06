from models.r18 import ResNet18
from torchensemble import VotingClassifier
from torchensemble.utils import io
import argparse
import os

import torch.nn as nn

from utils import DataLoad
from EvaluateMetrics import EvaluateROC, EvaluateUncertainty, EvaluateACC

parser = argparse.ArgumentParser(
    description='Train Ensemble ResNet18 for Strabismus Dataset')

parser.add_argument('--n_estimators', type=int, default=2, help='Number of estimators in ensemble')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--mode', type=str, default='Train',
                    help='Train/Test model')

parser.add_argument('--data_path', type=str, default='~/Datasets/TrainValidDataset')
parser.add_argument('--img_mode', type=str, default='full',
                    help='full/cropped img')

parser.add_argument('--ck_dir', type=str, default='../Checkpoint')
parser.add_argument('--net', type=str, default='ResNet18')

args = parser.parse_args()

args.ck_dir = os.path.join(args.ck_dir, 'ResNet18_{}'.format(args.n_estimators))  # 确保加载的模型estimators个数与当前一致

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)


def train_ensemble(DataTotal):
    # Initialize model
    model = VotingClassifier(
        estimator=ResNet18,
        n_estimators=args.n_estimators,
        cuda=True,
    )

    # Load Data
    train_loader, test_loader = DataTotal["Train"], DataTotal["Test"]

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    model.set_criterion(criterion)

    model.set_optimizer('Adam',
                        lr=1e-3,
                        weight_decay=5e-4)

    # Training
    model.fit(train_loader=train_loader,
              test_loader=test_loader,
              epochs=args.epochs,
              save_model=True,
              save_dir=args.ck_dir)

    return model


def test_ensemble(DataTotal):
    # Initialize model
    model = VotingClassifier(
        estimator=ResNet18,
        n_estimators=args.n_estimators,
        cuda=True,
    )

    # Load trained model
    io.load(model, save_dir=args.ck_dir)
    print("模型精确度：{}".format(EvaluateACC(model,DataTotal['Test'])))

    # Plot ROC Curve
    # EvaluateROC(model, DataTotal, args)

    # Evaluate Uncertainty, includes [std, entropy]
    EvaluateUncertainty(model, DataTotal)


if __name__ == "__main__":
    DataTotal = DataLoad(DatasetPath=args.data_path, batch_size=args.batch_size, img_mode=args.img_mode)
    if args.mode == 'Train':
        Ensemble_model = train_ensemble(DataTotal)
    elif args.mode == 'Test':
        test_ensemble(DataTotal)
    else:
        raise Exception("Unknown mode, please check!!")
