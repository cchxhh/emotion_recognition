from __future__ import annotations

from torch.utils.data import ConcatDataset, Subset


def split_dataset(dataset, *, folds=10):
    dataset_size = len(dataset)
    single_split_size = dataset_size // folds
    for i in range(folds):
        # 生成当前折的测试集和训练集的索引
        test_indices = list(range(i * single_split_size, (i + 1) * single_split_size))
        train_indices = list(range(0, i * single_split_size)) + list(range((i + 1) * single_split_size, dataset_size))

        # 根据索引创建数据子集
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        yield train_subset, test_subset


def split_dataset_by_class(dataset, num_classes=7, *, folds=10):
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_,label) in enumerate(dataset):
        #print(idx)
        class_indices[label].append(idx)
    for i in range(folds):
        class_train_subsets = []
        class_test_subsets = []
        for indices in class_indices:
            class_subset = Subset(dataset, indices)
            subset_size = len(class_subset)
            single_split_size = subset_size // folds

            test_indices = list(range(i * single_split_size, (i + 1) * single_split_size))
            train_indices = list(range(0, i * single_split_size)) + list(
                range((i + 1) * single_split_size, subset_size)
            )

            class_train_subsets.append(Subset(class_subset, train_indices))
            class_test_subsets.append(Subset(class_subset, test_indices))

        # 根据索引创建数据子集
        train_subset = ConcatDataset(class_train_subsets)
        test_subset = ConcatDataset(class_test_subsets)
        yield train_subset, test_subset
