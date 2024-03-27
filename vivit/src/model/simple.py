from __future__ import annotations

import torch
from torch import nn

from transformers import  VivitForVideoClassification,AutoModel

class vivitModel(nn.Module):
    def __init__(self, num_classes):
        super(vivitModel, self).__init__()

        # 加载预训练的 Vivit 模型
        #self.pre_model = VivitForVideoClassification.from_pretrained("/home/cv/Project1/cxh/multi_model/vivit/model/vivit-b-16x2-kinetics400", local_files_only=True)
        self.backbone = AutoModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        
        for name,param in self.backbone.named_parameters():
            #if not name.startswith('vivit.pooler'):
            param.requires_grad = False
        
        self.hidden_size = self.backbone.config.hidden_size
        #in_features = self.backbone.pooler.output_features
        #self.classifier = nn.Linear(hidden_size , num_classes)
        # torch.nn.init.xavier_uniform_(self.pre_model.classifier.weight)
        # torch.nn.init.zeros_(self.pre_model.classifier.bias)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, num_classes),
        )

    def forward(self, inputs):
        # 定义模型的前向传播
        x= self.backbone(inputs)
        features = x.pooler_output
        x=self.classifier(features)
        return x