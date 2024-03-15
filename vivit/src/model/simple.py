from __future__ import annotations

import torch
from torch import nn

from transformers import  VivitForVideoClassification

class vivitModel(nn.Module):
    def __init__(self, num_classes):
        super(vivitModel, self).__init__()

        # 加载预训练的 Vivit 模型
        self.pre_model = VivitForVideoClassification.from_pretrained("/home/cv/Project1/cxh/multi_model/vivit/model/vivit-b-16x2-kinetics400", local_files_only=True)
        for param in self.pre_model.parameters():
            param.requires_grad = False
        
        in_features = self.pre_model.classifier.in_features
        self.pre_model.classifier = torch.nn.Linear(in_features , num_classes)
        torch.nn.init.xavier_uniform_(self.pre_model.classifier.weight)
        torch.nn.init.zeros_(self.pre_model.classifier.bias)

    def forward(self, inputs):
        # 定义模型的前向传播
        x= self.pre_model(inputs)
        return x