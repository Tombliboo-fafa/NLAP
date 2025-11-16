import numpy as np
import torch
import torch.nn.functional as F

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

def val_covering_rate(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, _, partial_y, _, _ in loader:
            x = x.to(device)
            partial_y = partial_y.to(device)
            p = model(x)
            predicted_label = p.argmax(1)
            covering_per_example = partial_y[torch.arange(len(x)), predicted_label]
            correct += covering_per_example.sum().item()
            total += len(x)

    return correct / total

def val_approximated_accuracy(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, _, partial_y, _, _ in loader:
            x = x.to(device)
            partial_y = partial_y.to(device)
            batch_outputs = model(x)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            label_confidence = temp_un_conf * partial_y
            base_value = label_confidence.sum(dim=1).unsqueeze(1).repeat(1, label_confidence.shape[1]) + 1e-12
            label_confidence = label_confidence / base_value
            predicted_label = batch_outputs.argmax(1)
            risk_mat = torch.ones_like(partial_y).float()
            risk_mat[torch.arange(len(x)), predicted_label] = 0
            correct += len(x) - (risk_mat * label_confidence).sum().item()
            total += len(x)
    return correct / total

def evaluate(val_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, _, partial_y, y, _ in val_loader:
            data = data.to(device)
            partial_y = partial_y.to(device)
            y = y.to(device)
            logits = model(data)
            _, pred = torch.max(logits.data, 1)
            total += partial_y.size(0)
            correct += (pred == y.long()).sum()
        acc = float(correct) / float(total)
    return acc

def evaluate_test(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, y, _ in test_loader:
            data = data.to(device)
            y = y.to(device)
            logits = model(data)
            _, pred = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (pred == y.long()).sum()
        acc = float(correct) / float(total)
    return acc
