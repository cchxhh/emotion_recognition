from __future__ import annotations

import os

import torch
from loguru import logger
from rich.progress import Progress
from torch import nn
from torch.utils.data import DataLoader
import pickle

from transformers import  VivitForVideoClassification
from extractor.dataset.utils import split_dataset_by_class
from extractor.dataset.video import VideoFeaturesDataset
from extractor.utils import EarlyStopper, calculate_accuracy, calculate_f1_score, load_features
from model.simple import vivitModel


logger.add("logs/train_and_eval.log", rotation="10 MB")


def train_and_eval(
    model: nn.Module,
    num_epochs: int, 
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
):
#     optimizer = torch.optim.Adam([
#     {'params': fc.parameters(), 'lr': 0.001},  # 新的全连接层的学习率
#     {'params': pre_model.parameters()}  # 其他参数使用默认学习率
# ], lr=0.0001)
    #optimizer = torch.optim.Adam([{'params':model.classifier.parameters(), 'lr':0.0001},{'params':model.backbone.pooler.parameters(), 'lr':0.0001}])
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)

    with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns()) as progress:
        stopper = EarlyStopper(5)

        task = progress.add_task(
            f"[green]Begin to train",
            total=num_epochs,
            loss_value=float("inf"),
        )
        for epoch in range(num_epochs):
            model.train()
            loss_value = float("inf")
            loss_value_list = []
            for features, labels in train_data_loader:
                # print(features.shape)
                # print(labels.shape,len(labels))
                outputs=[]
                for i in range(len(labels)):
                    output=[]
                    optimizer.zero_grad()
                    #print(features[i].shape,labels)
                    #outputs = model(torch.tensor(features[i]))
                    output = model(features[i])
                    #output = output.logits
                    outputs.append(output)
                outputs = torch.cat(outputs,dim=0)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_value_list.append(loss.item())
            loss_value = sum(loss_value_list) / len(loss_value_list)
            progress.update(task, advance=1, loss_value=loss_value)
            if epoch % 10 == 0:
                test_accuracy = calculate_accuracy(model, test_data_loader)
                if stopper.update(loss=-loss_value, accuracy=test_accuracy):
                    break

    train_accuracy = calculate_accuracy(model, train_data_loader)
    test_accuracy = calculate_accuracy(model, test_data_loader)

    return train_accuracy, test_accuracy


num_epochs = 100
batch_size = 256
num_classes = 7
use_cuda = True
features_path = "/home/cv/Project1/cxh/multi_model/vivit/vivit/src/features/data.pkl"

criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
mean_accuracies = {}
mean_f1_scores = {}

# pre_model = VivitForVideoClassification.from_pretrained("/home/cv/Project1/cxh/multi_model/vivit/model/vivit-b-16x2-kinetics400",local_files_only=True)
# #print(model)
# for param in pre_model.parameters():
#     param.requires_grad = False

# new_features = pre_model.classifier.out_features
# fc = torch.nn.Linear(new_features,num_classes)
# model = torch.nn.Sequential(pre_model,fc)
# model = VivitForVideoClassification.from_pretrained("/home/cv/Project1/cxh/multi_model/vivit/model/vivit-b-16x2-kinetics400",local_files_only=True)

#print(model)


num_classes = 7 
model = vivitModel(num_classes)
print(model)
# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}")
#     print(f"Parameter shape: {param.shape}")
#     print(f"Parameter requires gradient: {param.requires_grad}")
#     print('---')

with open(features_path, "rb") as file:
        data = pickle.load(file)
features_dict = data
#print(features_dict)
mean_accuracy = 0
mean_f1_score = 0
for i, (train_dataset, test_dataset) in enumerate(
    split_dataset_by_class(VideoFeaturesDataset(features_dict, use_cuda=use_cuda), folds=10)
):
    
    checkpoint_dir = f"checkpoints/models/video/lr0.0001_dropout0.2_epoch100"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = f"{checkpoint_dir}/{i}.pth"

    # if os.path.exists(model_path):
    #     logger.info(f"skip: {model_path} exists")
    #     continue
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # m= next(iter(train_data_loader))
    # print(m[1].shape)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if use_cuda:
        model = model.cuda()
    train_accuracy, test_accuracy = train_and_eval(
        model,
        num_epochs,
        train_data_loader,
        test_data_loader,
    )

    torch.save(
        model,
        model_path,
    )
    logger.info(f"Model saved: {model_path}")

    mean_accuracy += test_accuracy
    logger.info(
        f"Accuracy in ({i+1}/10): test: {test_accuracy:.2f}%, train: {train_accuracy:.2f}%"
    )
    f1_score = calculate_f1_score(model, test_data_loader)
    mean_f1_score += f1_score
    logger.info(f"F1 Score: {f1_score:.2f}")
mean_accuracy /= 10
mean_f1_score /= 10
logger.info(f"Mean Accuracy : {mean_accuracy:.2f}%")


logger.info(mean_accuracy)
logger.info(mean_f1_score)
