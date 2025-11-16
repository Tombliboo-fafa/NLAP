import numpy as np
import torch.nn.functional as F
import torch
import resnet
from lenet import LeNet
from torch.utils.data import DataLoader, TensorDataset

def dataset_split(train_images, train_labels, dataset, partial_rate=0.3, partial_bias=0.3, split_per=0.9, random_seed=10):

    if dataset == 'mnist':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dealed_labels = get_instance_partial_label(dataname=dataset, n=partial_rate, b=partial_bias, dataset=data, labels=targets, num_classes=10,
                                                feature_size=784, norm_std=0.1, seed=random_seed)

    if dataset == 'kmnist':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dealed_labels = get_instance_partial_label(dataname=dataset, n=partial_rate, b=partial_bias, dataset=data, labels=targets, num_classes=10,
                                                feature_size=784, norm_std=0.1, seed=random_seed)

    if dataset == 'fmnist':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dealed_labels = get_instance_partial_label(dataname=dataset, n=partial_rate, b=partial_bias, dataset=data, labels=targets, num_classes=10,
                                                feature_size=784, norm_std=0.1, seed=random_seed)

    if dataset == 'cifar10':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dealed_labels = get_instance_partial_label(dataname=dataset, n=partial_rate, b=partial_bias, dataset=data, labels=targets, num_classes=10,
                                                feature_size=3072, norm_std=0.1, seed=random_seed)

    if dataset == 'cifar100':
        data = torch.from_numpy(train_images).float()
        targets = torch.from_numpy(train_labels)
        dealed_labels = get_instance_partial_label(dataname=dataset, n=partial_rate, b=partial_bias, dataset=data, labels=targets, num_classes=100,
                                                feature_size=3072, norm_std=0.1, seed=random_seed)

    dealed_labels = dealed_labels.squeeze()
    num_samples = int(dealed_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = dealed_labels[train_set_index], dealed_labels[val_set_index]
    train_ori, val_ori = np.array(targets)[train_set_index], np.array(targets)[val_set_index]

    train_labels = torch.from_numpy(train_labels)
    val_labels = torch.from_numpy(val_labels)
    train_ori = torch.from_numpy(train_ori)
    val_ori = torch.from_numpy(val_ori)

    return train_set, val_set, train_labels, val_labels, train_ori, val_ori

def gen_partial_label_by_p(probabilities, real_labels, n, b):
    # n -> partial_rate
    # b -> noise_rate
    num_rows, num_cols = probabilities.shape

    num_noisy_samples = int(num_rows * b)
    noisy_indices = torch.randperm(num_rows)[:num_noisy_samples]
    noisy_labels = real_labels.clone()
    for idx in noisy_indices:
        original_label = real_labels[idx].item()
        noisy_label = original_label
        while noisy_label == original_label:
            noisy_label = torch.randint(0, num_cols, (1,)).item()

        noisy_labels[idx] = noisy_label
        max_p = probabilities[idx, original_label].item()
        mean_p = (torch.sum(probabilities[idx])-max_p).item()
        mean_p = mean_p/(num_cols-1)
        probabilities[idx, noisy_label] = max_p
        probabilities[idx, original_label] = mean_p

    noisy_labels = F.one_hot(noisy_labels, num_classes=num_cols)

    train_p_Y = noisy_labels.clone().detach()
    probabilities[torch.where(noisy_labels == 1)] = 0
    probabilities = probabilities / torch.max(probabilities, dim=1, keepdim=True)[0]
    probabilities = probabilities / probabilities.mean(dim=1, keepdim=True) * n
    probabilities[probabilities > 1.0] = 1.0
    probabilities = torch.nan_to_num(probabilities, nan=0)
    m = torch.distributions.binomial.Binomial(total_count=1, probs=probabilities)
    z = m.sample()
    train_p_Y[torch.where(z == 1)] = 1.0

    avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
    return train_p_Y, avg_C.item()

def get_instance_partial_label(dataname, n, b, dataset, labels, num_classes, feature_size, norm_std, seed):

    if dataname == 'cifar10':
        weight_path = './trained_model/clean/cifar10.pt'
        model = resnet.ResNet18(input_channel=3, num_classes=10)
    elif dataname == 'mnist':
        weight_path = './trained_model/clean/mnist.pt'
        model = LeNet()
    elif dataname == 'kmnist':
        weight_path = './trained_model/clean/kmnist.pt'
        model = LeNet()
    elif dataname == 'fmnist':
        weight_path = './trained_model/clean/fmnist.pt'
        model = resnet.ResNet18(input_channel=1, num_classes=10)
    elif dataname == 'cifar100':
        weight_path = './trained_model/clean/cifar100.pt'
        model = resnet.ResNet50(input_channel=3, num_classes=100)

    current_device = torch.cuda.current_device()
    with ((torch.no_grad())):
        model = model.to(current_device)
        model.load_state_dict(torch.load(weight_path, map_location='cuda:'+str(current_device)))

        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed(int(seed))

        if dataname == 'cifar100' or dataname == 'cifar10':
            dataset = np.array(dataset)
            dataset = dataset.reshape((-1, 3, 32, 32))
            dataset = torch.tensor(dataset)

        data_tensor = TensorDataset(dataset, labels)
        dataloader = DataLoader(data_tensor, batch_size=128, shuffle=False, drop_last=False)

        P = []
        L = []
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(current_device)
            batch_labels = batch_labels.to(current_device)
            if dataname == 'cifar100' or dataname == 'cifar10':
                batch_pred = model(batch_data)
            else:
                batch_pred = model(batch_data.unsqueeze(1))
            P.append(batch_pred.clone().detach())
            L.append(batch_labels)

        P = torch.cat(P, dim=0).cpu()
        L = torch.cat(L, dim=0).cpu()

        if dataname == 'mnist' or dataname == 'kmnist':
            P = P / torch.norm(P, dim=1, keepdim=True)

        probs = F.softmax(P, dim=-1).clone().detach()
        train_p_Y, avg_c = gen_partial_label_by_p(probs, L, n, b)

    return np.array(train_p_Y)

def transform_target(label):
    target = label
    return target

def partial_loss(output1, target):
    logsm_outputs = F.log_softmax(output1, dim=1)
    final_outputs = logsm_outputs * target
    loss = - ((final_outputs).sum(dim=1)).mean()

    output = F.softmax(output1, dim=1)
    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    revisedY = revisedY * output
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)

    return loss, revisedY