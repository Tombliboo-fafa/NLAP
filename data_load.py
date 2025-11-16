import torch.utils.data as Data
from PIL import Image
import tools
import torch
import numpy as np


class mnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='mnist',
                 partial_rate=0.5, partial_bias=1.0, split_per=0.9, random_seed=1):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        original_images = np.load('data/benchmark/mnist/train_images.npy')
        original_labels = np.load('data/benchmark/mnist/train_labels.npy')

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_clean, self.val_clean = (
            tools.dataset_split(original_images,
                                original_labels, dataset,
                                partial_rate, partial_bias, split_per,
                                random_seed))

        self.train_data = self.train_data.reshape(len(self.train_data), 28, 28)
        self.val_data = self.val_data.reshape(len(self.val_data), 28, 28)
        self.train_data = torch.tensor(self.train_data, dtype=torch.uint8)
        self.val_data = torch.tensor(self.val_data, dtype=torch.uint8)

        train_row_sums = self.train_labels.sum(axis=1, keepdims=True)
        train_row_sums[train_row_sums == 0] = 1
        val_row_sums = self.val_labels.sum(axis=1, keepdims=True)
        val_row_sums[val_row_sums == 0] = 1
        self.train_labels_01 = self.train_labels.clone()
        self.val_labels_01 = self.val_labels.clone()
        self.train_labels = self.train_labels / train_row_sums
        self.val_labels = self.val_labels / val_row_sums

    def __getitem__(self, index):

        if self.train:
            img, label, label_01, clean = self.train_data[index], self.train_labels[index], self.train_labels_01[index], self.train_clean[index]
        else:
            img, label, label_01, clean = self.val_data[index], self.val_labels[index], self.val_labels_01[index], self.val_clean[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label_01, clean, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class mnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/benchmark/mnist/test_images.npy')
        self.test_labels = np.load('data/benchmark/mnist/test_labels.npy')
        self.test_data = self.test_data.reshape(len(self.test_data), 28, 28)
        self.test_data = torch.tensor(self.test_data, dtype=torch.uint8)

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)

class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='cifar10',
                 partial_rate=0.5, partial_bias=1.0, split_per=0.9, random_seed=1):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/benchmark/cifar10/train_images.npy')
        original_labels = np.load('data/benchmark/cifar10/train_labels.npy')

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_clean, self.val_clean = (
            tools.dataset_split(original_images,
                                original_labels, dataset,
                                partial_rate, partial_bias, split_per,
                                random_seed))

        if self.train:
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

        train_row_sums = self.train_labels.sum(axis=1, keepdims=True)
        train_row_sums[train_row_sums == 0] = 1
        val_row_sums = self.val_labels.sum(axis=1, keepdims=True)
        val_row_sums[val_row_sums == 0] = 1
        self.train_labels_01 = self.train_labels.clone()
        self.val_labels_01 = self.val_labels.clone()
        self.train_labels = self.train_labels / train_row_sums
        self.val_labels = self.val_labels / val_row_sums

    def __getitem__(self, index):

        if self.train:
            img, label, label_01, clean = self.train_data[index], self.train_labels[index], self.train_labels_01[index], \
            self.train_clean[index]
        else:
            img, label, label_01, clean = self.val_data[index], self.val_labels[index], self.val_labels_01[index], \
            self.val_clean[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label_01, clean, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class cifar10_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/benchmark/cifar10/test_images.npy')
        self.test_labels = np.load('data/benchmark/cifar10/test_labels.npy')
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)

class cifar100_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='cifar100',
                 partial_rate=0.5, partial_bias=1.0, split_per=0.9, random_seed=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/benchmark/cifar100/train_images.npy')
        original_labels = np.load('data/benchmark/cifar100/train_labels.npy')

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_clean, self.val_clean = (
            tools.dataset_split(original_images,
                                original_labels, dataset,
                                partial_rate, partial_bias, split_per,
                                random_seed))

        if self.train:
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

        train_row_sums = self.train_labels.sum(axis=1, keepdims=True)
        train_row_sums[train_row_sums == 0] = 1
        val_row_sums = self.val_labels.sum(axis=1, keepdims=True)
        val_row_sums[val_row_sums == 0] = 1
        self.train_labels_01 = self.train_labels.clone()
        self.val_labels_01 = self.val_labels.clone()
        self.train_labels = self.train_labels / train_row_sums
        self.val_labels = self.val_labels / val_row_sums

    def __getitem__(self, index):

        if self.train:
            img, label, label_01, clean = self.train_data[index], self.train_labels[index], self.train_labels_01[index], \
            self.train_clean[index]
        else:
            img, label, label_01, clean = self.val_data[index], self.val_labels[index], self.val_labels_01[index], \
            self.val_clean[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label_01, clean, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class cifar100_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/benchmark/cifar100/test_images.npy')
        self.test_labels = np.load('data/benchmark/cifar100/test_labels.npy')
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)

class fmnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='fmnist',
                 partial_rate=0.5, partial_bias=1.0, split_per=0.9, random_seed=1):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        original_images = np.load('data/benchmark/fmnist/train_images.npy')
        original_labels = np.load('data/benchmark/fmnist/train_labels.npy')

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_clean, self.val_clean = (
            tools.dataset_split(original_images,
                                original_labels, dataset,
                                 partial_rate, partial_bias, split_per,
                                 random_seed))

        self.train_data = self.train_data.reshape(len(self.train_data), 28, 28)
        self.val_data = self.val_data.reshape(len(self.val_data), 28, 28)
        self.train_data = torch.tensor(self.train_data, dtype=torch.uint8)
        self.val_data = torch.tensor(self.val_data, dtype=torch.uint8)

        train_row_sums = self.train_labels.sum(axis=1, keepdims=True)
        train_row_sums[train_row_sums == 0] = 1
        val_row_sums = self.val_labels.sum(axis=1, keepdims=True)
        val_row_sums[val_row_sums == 0] = 1
        self.train_labels_01 = self.train_labels.clone()
        self.val_labels_01 = self.val_labels.clone()
        self.train_labels = self.train_labels / train_row_sums
        self.val_labels = self.val_labels / val_row_sums

    def __getitem__(self, index):

        if self.train:
            img, label, label_01, clean = self.train_data[index], self.train_labels[index], self.train_labels_01[index], \
            self.train_clean[index]
        else:
            img, label, label_01, clean = self.val_data[index], self.val_labels[index], self.val_labels_01[index], \
            self.val_clean[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label_01, clean, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class fmnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/benchmark/fmnist/test_images.npy')
        self.test_labels = np.load('data/benchmark/fmnist/test_labels.npy')

        self.test_data = self.test_data.reshape(len(self.test_data), 28, 28)
        self.test_data = torch.tensor(self.test_data, dtype=torch.uint8)

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)

class kmnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='kmnist',
                 partial_rate=0.5, partial_bias=1.0, split_per=0.9, random_seed=1):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        original_images = np.load('data/benchmark/kmnist/train_images.npy')
        original_labels = np.load('data/benchmark/kmnist/train_labels.npy')

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_clean, self.val_clean = (
            tools.dataset_split(original_images,
                                original_labels, dataset,
                                partial_rate, partial_bias, split_per,
                                random_seed))

        self.train_data = self.train_data.reshape(len(self.train_data), 28, 28)
        self.val_data = self.val_data.reshape(len(self.val_data), 28, 28)
        self.train_data = torch.tensor(self.train_data, dtype=torch.uint8)
        self.val_data = torch.tensor(self.val_data, dtype=torch.uint8)

        train_row_sums = self.train_labels.sum(axis=1, keepdims=True)
        train_row_sums[train_row_sums == 0] = 1
        val_row_sums = self.val_labels.sum(axis=1, keepdims=True)
        val_row_sums[val_row_sums == 0] = 1
        self.train_labels_01 = self.train_labels.clone()
        self.val_labels_01 = self.val_labels.clone()
        self.train_labels = self.train_labels / train_row_sums
        self.val_labels = self.val_labels / val_row_sums

    def __getitem__(self, index):

        if self.train:
            img, label, label_01, clean = self.train_data[index], self.train_labels[index], self.train_labels_01[index], \
                self.train_clean[index]
        else:
            img, label, label_01, clean = self.val_data[index], self.val_labels[index], self.val_labels_01[index], \
                self.val_clean[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label_01, clean, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class kmnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/benchmark/kmnist/test_images.npy')
        self.test_labels = np.load('data/benchmark/kmnist/test_labels.npy')
        self.test_data = self.test_data.reshape(len(self.test_data), 28, 28)
        self.test_data = torch.tensor(self.test_data, dtype=torch.uint8)

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)