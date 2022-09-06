# Uncertainty Analysis using Ensemble method on Strabismus Dataset

## QuickStart

### Training

- Training on *Full Eyes Dataset*：

```shell
python main.py --n_estimators=[num_nets] --mode=Train
```

- Training on *Cropped Eyes Dataset*  **without** using Expand2Square:

```shell
python main.py --n_estimators=[num_nets]  --data_path=~/Datasets/Cropped --ck_dir=./Checkpoint/CroppedDataset
```

- Training on *Cropped Eyes Dataset*  **with** using Expand2Square

```shell
python main.py --n_estimators=[num_nets]  --img_mode=cropped --data_path=~/Datasets/Cropped --ck_dir=./Checkpoint/CroppedDataset/Expand2Square
```

### Test

Same as mentioned above, but remember to add `--mode=Test`

## TODO List

- [x] 在FullEyesSet 上训练E3ResNet18和E5ResNet18

- [x] 在CropEyesSet 上使用Expand2Square训练E2ResNet18
- [x] 在CropEyesSet 上使用DirectlyResize训练E2ResNet18
- [ ] 实现TTAUG 方法
- [ ] 对分类正确和错误图片，其网络的T个输出的分布做分析，验证是否正确分类的概率分布较为集中，错误分类的概率分布较为分散
- [ ] 分析加入TTAUG 方法后，模型ROC 曲线变化
- [ ] 分析加入Ensemble 方法后，与单个网络模型的ROC曲线变化
- [ ] 分析TTAUG 变换样例个数T 与模型ROC-AUC 之间的关系
- [ ] 分析Ensemble 网络集成个数N 与模型ROC-AUC 之间的关系