import torch
import models_jw
from utils import *
from tqdm import tqdm
import numpy as np
from configs import paramsuper, getters
import matplotlib.pyplot as plt
import torch.nn as nn


args = paramsuper.ICIFARHashResNet18()
kwargs = {'num_workers': 1, 'pin_memory': True}

def entropy(probs):
    eps = 1e-9
    probs = torch.clamp(probs, min=eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)

def plot_hists(listt, task_val_num = 9, smallheads=False, device = "cuda:4"):
    torch.manual_seed(1)
    np.random.seed(0)
    _, axs = plt.subplots(task_val_num,len(listt), sharey=True, figsize=(20,10))
    for idx, path in  tqdm(enumerate(listt)):
        kwargs = {'num_workers': 1, 'pin_memory': True, 'smallheads': smallheads}
        heads = 10 if smallheads else 100
        net2 = models_jw.HashResNet18(heads).to(device)
        net2.load_state_dict(torch.load(f"exp_ebms/{path}.pth"))

        test_loader = getters.get_dataset(args.dataset, 1, 1000, False, kwargs)
        tasks_num = 10
        s = nn.Softmax(dim=1)
        accs =  {i: 0 for i in range(tasks_num)}
        pp, ee, ys  =[], [], []
        for task_num in range(tasks_num):
            X, y = test_loader.get_data()
            X, y = X.to(device), y.to(device)
            p, e = [], []
            for ti in range(task_val_num):       
                z = torch.zeros(10, device=device)
                z[ti] = 1
                y_hat, _, _ = net2(X, z)
                p.append(y_hat.detach().cpu())
                e.append(entropy(s(-y_hat.detach().cpu())))
            pp.append(p)
            ee.append(e)
            ys.append(y.detach().cpu())
        
        y_preds = []
        buff = []
        for k in range(task_val_num):
            mins, mins_idxs=[],[]
            for i in range(task_val_num):
                mins.append(pp[k][i].min(1).values)
                mins_idxs.append(pp[k][i].min(1).indices)
            predicted_task_num = torch.stack(mins).min(0).indices
            buff.append(torch.stack(mins))
            y_pred, task_pred = 0, 0
            for ii in range(1000):
                if predicted_task_num[ii] == k:
                    task_pred+=1
                if (torch.stack(mins_idxs)[predicted_task_num[ii], ii] == ys[k][ii]) and  (predicted_task_num[ii] == k):
                    y_pred += 1  
            y_preds.append(y_pred)
            axs[k,idx].hist(predicted_task_num,bins=8)
            y_label = "ZERO" if task_pred == 0 else np.round(y_pred/task_pred*100,0)
            axs[k,idx].set_ylabel(y_label)
        axs[0,idx].set_title(f"{path},{sum(y_preds)}")
     #   return buff

def check_accs(path, smallheads=True, device="cuda:4"):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'smallheads': smallheads}
    heads = 10 if smallheads else 100
    net2 = models_jw.HashResNet18(heads).to(device)
    net2.load_state_dict(torch.load(f"exp_ebms/{path}.pth"))
    net2.to(device)

    test_loader = getters.get_dataset(args.dataset, 1, 1000, False, kwargs)
    tasks_num = 10
    accs =  {i: 0 for i in range(tasks_num)}
    for time in tqdm(range(tasks_num)):
        X, y = test_loader.get_data()
        X, y = X.to(device), y.to(device)        
        z = torch.zeros(10, device=device)
        z[time] = 1
        y_hat, _, _ = net2(X, z)
        accs[time] = accs[time] + (y_hat.min(1).indices==y).sum().item()
    return accs

