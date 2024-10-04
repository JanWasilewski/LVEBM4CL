import torch
import pickle
import models_jw
from utils import *
from tqdm import tqdm
import numpy as np
from configs import paramsuper, getters

args = paramsuper.ICIFARHashResNet18()

torch.manual_seed(args.seed)
np.random.seed(0)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True, "smallheads": False} 

margins = [
    [100]*8,
    [200]*8,
    [300]*8,
    range(100, 900, 100),
    range(100, 500, 50),
    range(450, 50, -50),
    range(800, 0, -100)
]

for fuzzy in [0,1]:
    for hingeLoss in range(1,3):
        for idxs, margin in enumerate(margins):
            torch.manual_seed(args.seed)
            np.random.seed(0)
            TASKS_NUM = 8
            net = models_jw.HashResNet18(100).to(device)
            train_loader = getters.get_dataset(args.dataset, args.period, args.batch_size, True, kwargs)
            test_loader = getters.get_dataset(args.dataset, 1, args.test_batch_size, False, kwargs)
            optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
            diffs = []
            for time in tqdm(range(TASKS_NUM*args.period)):
                task_idx = time // args.period
                X, y = train_loader.get_data()
                X, y = X.to(device), y.to(device)        
                z = prepare_z(task_idx, device, fuzzy)
                y_hat, _, _ = net(X, z)
                optimizer.zero_grad()
                loss = hinge_loss(y_hat, y, margin[task_idx], type=hingeLoss, task_num=task_idx)
                loss.backward()
                optimizer.step()
                diffs.append(torch.mean(y_hat.max(1).values - y_hat.min(1).values).item())

            with open(f"ebms/{hingeLoss}_{idxs}_{fuzzy}_0_bh", "wb") as fp:   
                pickle.dump(diffs, fp)
            torch.save(net.state_dict(), f"ebms/{hingeLoss}_{idxs}_{fuzzy}_0_bh.pth")