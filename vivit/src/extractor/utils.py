from __future__ import annotations

import pickle

import cv2
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader


def get_example_image():
    dataset_path = "./datasets"
    input_image = cv2.imread(f"{dataset_path}/Face-Dataset/UCEC-Face/subject1/subject1.4.png")
    return input_image


def load_features(features_path: str, name: str):
    file_path = f"{features_path}/{name}.pkl"
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def calculate_accuracy(model: nn.Module, data_loader: DataLoader):
    # model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in data_loader:
            for i in range(len(labels)):
                outputs = model(features[i])
                _, predicted = torch.max(outputs.logits.data, 1)  # 获取最大概率的预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测的样本数

    accuracy = 100 * correct / total
    return accuracy


def calculate_class_weights(data_loader: DataLoader, num_classes: int = 130):
    class_counts = [0] * num_classes

    for _, labels in data_loader:
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum().item()

    total_samples = sum(class_counts)
    class_weights = []
    for i in range(num_classes):
        class_weights.append(class_counts[i] / total_samples)

    return class_weights


def calculate_f1_score(model: nn.Module, data_loader: DataLoader):
    class_weights = calculate_class_weights(data_loader)
    num_classes = len(class_weights)
    class_f1_scores = [0] * num_classes
    class_counts = [0] * num_classes

    for features, labels in data_loader:
        for i in range(len(labels)):
            outputs = model(features[i])
            _, predicted = torch.max(outputs.logits.data, 1)

        for i in range(num_classes):
            true_positives = ((predicted == i) & (labels == i)).sum().item()
            false_positives = ((predicted == i) & (labels != i)).sum().item()
            false_negatives = ((predicted != i) & (labels == i)).sum().item()

            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)

            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

            class_f1_scores[i] += f1_score
            class_counts[i] += 1

    weighted_f1_score = 0

    for i in range(num_classes):
        class_weight = class_weights[i]
        class_f1 = class_f1_scores[i] / class_counts[i]
        weighted_f1_score += class_weight * class_f1
    return 100 * weighted_f1_score


class EarlyStopper:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_scores = {}

    def update(self, **kwargs: float):
        for key, value in kwargs.items():
            if key not in self.best_scores:
                self.best_scores[key] = (float("-inf"), 0)
            if value > self.best_scores[key][0]:
                self.best_scores[key] = (value, self.best_scores[key][1])
            else:
                self.best_scores[key] = (self.best_scores[key][0], self.best_scores[key][1] + 1)
                if self.best_scores[key][1] >= self.patience:
                    logger.info(f"Early stopping by {key}!")
                    return True
        return False

def calculate_accuracy_vl(model: nn.Module, data_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels,lengths in data_loader:
            outputs = model(features,lengths)
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测的样本数

    accuracy = 100 * correct / total
    return accuracy


def calculate_class_weights_vl(data_loader: DataLoader, num_classes: int = 130):
    class_counts = [0] * num_classes

    for _, labels,_ in data_loader:
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum().item()

    total_samples = sum(class_counts)
    class_weights = []
    for i in range(num_classes):
        class_weights.append(class_counts[i] / total_samples)

    return class_weights


def calculate_f1_score_vl(model: nn.Module, data_loader: DataLoader):
    class_weights = calculate_class_weights_vl(data_loader)
    num_classes = len(class_weights)
    class_f1_scores = [0] * num_classes
    class_counts = [0] * num_classes

    for features, labels,lengths in data_loader:
        outputs = model(features,lengths)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(num_classes):
            true_positives = ((predicted == i) & (labels == i)).sum().item()
            false_positives = ((predicted == i) & (labels != i)).sum().item()
            false_negatives = ((predicted != i) & (labels == i)).sum().item()

            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)

            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

            class_f1_scores[i] += f1_score
            class_counts[i] += 1

    weighted_f1_score = 0

    for i in range(num_classes):
        class_weight = class_weights[i]
        class_f1 = class_f1_scores[i] / class_counts[i]
        weighted_f1_score += class_weight * class_f1
    return 100 * weighted_f1_score