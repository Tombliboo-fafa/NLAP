import argparse
import random
import warnings
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import data_load
import resnet
import tools
from lenet import LeNet
import torch.nn.functional as F
import misc
import time
import psutil

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=int, default=1, help="GPU ID")
parser.add_argument('--partial_rate', type=float, help='overall corruption rate, should be less than 1', default=0.3)
parser.add_argument('--partial_bias', type=float, help='label sharpness, the value should be no less than 0, the bigger the sharper', default=0.3)
parser.add_argument('--partial_type', type=str, help='[clean, instance, generalized_instance]', default='instance')
parser.add_argument('--dataset', type=str, help='[mnist, fmnist, kmnist, cifar10, cifar100]', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=60)
parser.add_argument('--warm_epoch', type=int, default=5)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--weight_decay', type=float, help='l2', default=1e-5)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
parser.add_argument('--tips', type=str, help='tips', default='UIDPLL')

parser.add_argument('--thr_add', type=float, default=0.7)
parser.add_argument('--k', type=int, default=30)

args = parser.parse_args()

torch.cuda.set_device(args.gpu)
device = torch.device('cuda:' + str(args.gpu))
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore')

def load_data(args):

    if args.dataset == 'kmnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.kmnist_dataset(True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,)), ]),
                                                 target_transform=tools.transform_target,
                                                 dataset=args.dataset,
                                                 partial_rate=args.partial_rate,
                                                 partial_bias=args.partial_bias,
                                                 split_per=args.split_percentage,
                                                 random_seed=args.seed)

        val_dataset = data_load.kmnist_dataset(False,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,), (0.5,)), ]),
                                               target_transform=tools.transform_target,
                                               dataset=args.dataset,
                                               partial_rate=args.partial_rate,
                                               partial_bias=args.partial_bias,
                                               split_per=args.split_percentage,
                                               random_seed=args.seed)

        test_dataset = data_load.kmnist_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)), ]),
            target_transform=tools.transform_target)

    if args.dataset == 'fmnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.fmnist_dataset(True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                 target_transform=tools.transform_target,
                                                 dataset=args.dataset,
                                                 partial_rate=args.partial_rate,
                                                 partial_bias=args.partial_bias,
                                                 split_per=args.split_percentage,
                                                 random_seed=args.seed)

        val_dataset = data_load.fmnist_dataset(False,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,)), ]),
                                               target_transform=tools.transform_target,
                                               dataset=args.dataset,
                                               partial_rate=args.partial_rate,
                                               partial_bias=args.partial_bias,
                                               split_per=args.split_percentage,
                                               random_seed=args.seed)

        test_dataset = data_load.fmnist_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)), ]),
            target_transform=tools.transform_target)

    if args.dataset == 'mnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.mnist_dataset(True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                partial_rate=args.partial_rate,
                                                partial_bias=args.partial_bias,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)
        val_dataset = data_load.mnist_dataset(False,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,)),
                                              ]),
                                              target_transform=tools.transform_target,
                                              dataset=args.dataset,
                                              partial_rate=args.partial_rate,
                                              partial_bias=args.partial_bias,
                                              split_per=args.split_percentage,
                                              random_seed=args.seed)

        test_dataset = data_load.mnist_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
            target_transform=tools.transform_target)

    if args.dataset == 'cifar10':
        args.channel = 3
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar10_dataset(True,
                                                  transform=transforms.Compose([
                                                      transforms.RandomCrop(32, padding=4),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                           (0.2023, 0.1994, 0.2010)),
                                                  ]),
                                                  target_transform=tools.transform_target,
                                                  dataset=args.dataset,
                                                  partial_rate=args.partial_rate,
                                                  partial_bias=args.partial_bias,
                                                  split_per=args.split_percentage,
                                                  random_seed=args.seed)

        val_dataset = data_load.cifar10_dataset(False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                partial_rate=args.partial_rate,
                                                partial_bias=args.partial_bias,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)

        test_dataset = data_load.cifar10_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            target_transform=tools.transform_target)

    if args.dataset == 'cifar100':
        args.channel = 3
        args.num_classes = 100
        args.feature_size = 3 * 32 * 32
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar100_dataset(True,
                                                   transform=transforms.Compose([
                                                       transforms.RandomCrop(32, padding=4),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2023, 0.1994, 0.2010)),
                                                   ]),
                                                   target_transform=tools.transform_target,
                                                   dataset=args.dataset,
                                                   partial_rate=args.partial_rate,
                                                   partial_bias=args.partial_bias,
                                                   split_per=args.split_percentage,
                                                   random_seed=args.seed)

        val_dataset = data_load.cifar100_dataset(False,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010)),
                                                 ]),
                                                 target_transform=tools.transform_target,
                                                 dataset=args.dataset,
                                                 partial_rate=args.partial_rate,
                                                 partial_bias=args.partial_bias,
                                                 split_per=args.split_percentage,
                                                 random_seed=args.seed)

        test_dataset = data_load.cifar100_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            target_transform=tools.transform_target)

    return train_dataset, val_dataset, test_dataset

def compute_knn(features, k=10, block_size=10):
    N, D = features.size()
    neighbors = torch.zeros((N, k), dtype=torch.long, device=features.device)

    for i in range(0, N, block_size):
        end_i = min(i + block_size, N)
        distances_block = []

        for j in range(0, N, block_size):
            end_j = min(j + block_size, N)
            block_distances = torch.cdist(features[i:end_i], features[j:end_j], p=2)
            distances_block.append(block_distances)

        distances_block = torch.cat(distances_block, dim=1)
        knn_indices_block = torch.argsort(distances_block, dim=1)[:, 1: k + 1]
        neighbors[i:end_i] = knn_indices_block
    return neighbors

def loss_PLL(outpu, labels):
    partial_loss, new_target = tools.partial_loss(outpu, labels)
    return partial_loss, new_target

def warm_up(train_loader, model_NLAP, optimizer_NLAP):
    model_NLAP.train()

    train_total = 0
    train_correct = 0
    train_loss = []

    for i, (data, labels, _, clean_label, indexes) in enumerate(train_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()
            clean_label = clean_label.cuda()

        output = model_NLAP(data)

        partial_loss, new_label = loss_PLL(output, labels)

        loss = partial_loss
        train_loss.append(loss.item())

        pred = torch.max(output, 1)[1]
        correct_num = (pred == clean_label).sum()
        train_correct += correct_num.item()
        train_total += data.size(0)

        optimizer_NLAP.zero_grad()
        loss.backward()
        optimizer_NLAP.step()

        for j, k in enumerate(indexes):
            train_loader.dataset.train_labels[k, :] = new_label[j, :].detach()

    train_acc = float(train_correct) / float(train_total)

    return train_acc, train_loss

def update_labels(pred_values, neighbors, labels, thr_add):
    predicted_labels = pred_values.argmax(dim=1)
    pred_values = F.softmax(pred_values)
    labels_add = (labels == 0)
    new_labels = labels.clone()
    for i in range(pred_values.shape[0]):
        neighbor_idxs = neighbors[i]
        mask_same_label = predicted_labels[neighbor_idxs] == predicted_labels[i]
        same_label_neighbors = neighbor_idxs[mask_same_label]
        if len(same_label_neighbors) > 0:
            neighbors_labels = torch.sum(pred_values[same_label_neighbors], dim=0) / len(same_label_neighbors)

            if labels_add[i].sum() > 0:
                neighbors_labels_add = torch.where(neighbors_labels > thr_add, torch.tensor(1),
                                                   torch.tensor(0))
                added_label = neighbors_labels_add & labels_add[i]
                new_labels[i] = new_labels[i] + added_label * torch.mean(labels[i][labels[i] != 0])
                new_labels[i] = new_labels[i] / torch.sum(new_labels[i])

    return new_labels

def train_UIDPLL(train_loader, epoch, model_NLAP, optimizer_NLAP, args):
    model_NLAP.train()

    train_total = 0
    train_correct = 0
    train_loss = []

    thr_step = -0.01

    for i, (data, labels, _, clean_label, indexes) in enumerate(train_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()
            clean_label = clean_label.cuda()

        outpu = model_NLAP(data)
        loss, new_label = loss_PLL(outpu, labels)

        train_loss.append(loss.item())

        pred = torch.max(outpu, 1)[1]
        correct_num = (pred == clean_label).sum()
        train_correct += correct_num.item()
        train_total += data.size(0)

        optimizer_NLAP.zero_grad()
        loss.backward()
        optimizer_NLAP.step()

        for j, k in enumerate(indexes):
            train_loader.dataset.train_labels[k, :] = new_label[j, :].detach()

    for i, (data, labels, _, _, indexes) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            features = model_NLAP.extract_features(data).detach()
        neighbors = compute_knn(features, k=args.k)

        thr_add_now = max(0, min(1, args.thr_add + thr_step * epoch))

        with torch.no_grad():
            pred_values = model_NLAP(data).detach()

        new_labels = update_labels(pred_values, neighbors, labels, thr_add=thr_add_now)

        for j, k in enumerate(indexes):
            train_loader.dataset.train_labels[k, :] = new_labels[j, :].detach()

    train_acc = float(train_correct) / float(train_total)

    return train_acc, train_loss

def main(args):
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    train_dataset, val_dataset, test_dataset = load_data(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    learning_rate = args.lr
    if args.dataset == 'mnist':
        model_NLAP = LeNet()
        optimizer_NLAP = torch.optim.Adam(model_NLAP.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer_NLAP, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'kmnist':
        num_classes = 10
        model_NLAP = resnet.ResNet18(input_channel=1, num_classes=num_classes)
        optimizer_NLAP = torch.optim.Adam(model_NLAP.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer_NLAP, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'fmnist':
        num_classes = 10
        model_NLAP = resnet.ResNet18(input_channel=1, num_classes=num_classes)
        optimizer_NLAP = torch.optim.Adam(model_NLAP.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer_NLAP, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'cifar10':
        num_classes = 10
        model_NLAP = resnet.ResNet18(input_channel=3, num_classes=num_classes)
        optimizer_NLAP = torch.optim.Adam(model_NLAP.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer_NLAP, milestones=[40, 80], gamma=0.1)
    elif args.dataset == 'cifar100':
        num_classes = 100
        model_NLAP = resnet.ResNet50(input_channel=3, num_classes=num_classes)
        optimizer_NLAP = torch.optim.Adam(model_NLAP.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer_NLAP, milestones=[40, 80], gamma=0.1)

    if torch.cuda.is_available():
        model_NLAP.cuda()

    test_accuracy_curve = []
    val_accuracy_curve = []
    val_approxim_curve = []
    val_covering_curve = []
    loss_curve = []

    best_aa = 0.0
    best_test_aa = 0.0
    best_oa = 0.0
    best_test_oa = 0.0
    best_cr = 0.0
    best_test_cr = 0.0

    # warm up
    last_results_keys = None
    for warm_epoch in range(0, args.warm_epoch):
        results = {'epoch_warm': warm_epoch + 1}

        _, warm_loss = warm_up(train_loader, model_NLAP, optimizer_NLAP)

        val_acc = misc.evaluate(val_loader, model_NLAP, device)
        test_acc = misc.evaluate_test(test_loader, model_NLAP, device)
        val_covering_rate = misc.val_covering_rate(model_NLAP, val_loader, device)
        val_approximated_acc = misc.val_approximated_accuracy(model_NLAP, val_loader, device)
        results['lr'] = optimizer_NLAP.state_dict()['param_groups'][0]['lr']
        results['loss'] = warm_loss[0]
        results['test_acc'] = test_acc
        results['val_accuracy'] = val_acc
        results['val_approximated_acc'] = val_approximated_acc
        results['val_covering_rate'] = val_covering_rate
        results_keys = sorted(results.keys())
        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys], colwidth=12)

        test_accuracy_curve.append(results['test_acc'])
        val_accuracy_curve.append(results['val_accuracy'])
        val_approxim_curve.append(results['val_approximated_acc'])
        val_covering_curve.append(results['val_covering_rate'])
        loss_curve.append(warm_loss)

        if results['val_accuracy'] > best_oa:
            best_oa = results['val_accuracy']
            best_test_oa = results['test_acc']
        if results['val_approximated_acc'] > best_aa:
            best_aa = results['val_approximated_acc']
            best_test_aa = results['test_acc']
        if results['val_covering_rate'] > best_cr:
            best_cr = results['val_covering_rate']
            best_test_cr = results['test_acc']

    print(f"\nBest test Accuracy(warmup) by OA: {best_test_oa:.4f}")
    print(f"\nBest test Accuracy(warmup) by CR: {best_test_cr:.4f}")
    print(f"\nBest test Accuracy(warmup) by AA: {best_test_aa:.4f}")

    best_aa = 0.0
    best_test_aa = 0.0
    best_oa = 0.0
    best_test_oa = 0.0
    best_cr = 0.0
    best_test_cr = 0.0

    last_results_keys = None

    epoch_times = []
    gpu_peaks = []
    cpu_mems = []

    for epoch in range(0, args.n_epoch):
        process = psutil.Process()
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        scheduler.step()

        results = {'epoch': epoch + 1}

        _, train_loss = train_UIDPLL(train_loader, epoch, model_NLAP, optimizer_NLAP, args)

        end_time = time.time()
        epoch_time = end_time - start_time
        gpu_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB
        cpu_mem = process.memory_info().rss / 1024 ** 2  # MB

        epoch_times.append(epoch_time)
        gpu_peaks.append(gpu_peak)
        cpu_mems.append(cpu_mem)

        print(f"Epoch [{epoch + 1}/{args.n_epoch}] | Time: {epoch_time:.2f}s | "
              f"GPU Peak: {gpu_peak:.2f} MB | CPU: {cpu_mem:.2f} MB")

        val_acc = misc.evaluate(val_loader, model_NLAP, device)
        test_acc = misc.evaluate_test(test_loader, model_NLAP, device)
        val_covering_rate = misc.val_covering_rate(model_NLAP, val_loader, device)
        val_approximated_acc = misc.val_approximated_accuracy(model_NLAP, val_loader, device)
        results['lr'] = optimizer_NLAP.state_dict()['param_groups'][0]['lr']
        results['loss_all'] = train_loss[0]
        results['test_acc'] = test_acc
        results['val_accuracy'] = val_acc
        results['val_approximated_acc'] = val_approximated_acc
        results['val_covering_rate'] = val_covering_rate

        results_keys = sorted(results.keys())
        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys], colwidth=12)

        test_accuracy_curve.append(results['test_acc'])
        val_accuracy_curve.append(results['val_accuracy'])
        val_approxim_curve.append(results['val_approximated_acc'])
        val_covering_curve.append(results['val_covering_rate'])
        loss_curve.append(train_loss)
        if results['val_accuracy'] > best_oa:
            best_oa = results['val_accuracy']
            best_test_oa = results['test_acc']
        if results['val_approximated_acc'] > best_aa:
            best_aa = results['val_approximated_acc']
            best_test_aa = results['test_acc']
        if results['val_covering_rate'] > best_cr:
            best_cr = results['val_covering_rate']
            best_test_cr = results['test_acc']

    print(f"\nBest test Accuracy by OA: {best_test_oa:.4f}")
    print(f"\nBest test Accuracy by CR: {best_test_cr:.4f}")
    print(f"\nBest test Accuracy by AA: {best_test_aa:.4f}")

    return best_test_oa, best_test_aa, best_test_cr


if __name__ == '__main__':
    best_test_oa, best_test_aa, best_test_cr = main(args)
