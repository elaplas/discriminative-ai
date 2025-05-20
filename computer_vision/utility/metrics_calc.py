import torch

def convertLabelToProbability(labels):
    x = torch.zeros((labels.shape[0], 10)).float()
    for i in range(labels.shape[0]):
        x[i][labels[i]] = 1.0
    return x

def cal_accuracy(x: torch.tensor, y: torch.tensor):
    res = 0
    for i in range(x.size()[0]):
        if sum(abs(x[i] - y[i])) <= 0.1:
            res += 1
    res /= float(x.size()[0])
    return res