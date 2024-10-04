import torch
import pickle
import models_jw
from utils import *
from tqdm import tqdm
import numpy as np
from configs import paramsuper, getters

args = paramsuper.ICIFARHashResNet18()
args.period = 30000
torch.manual_seed(args.seed)
np.random.seed(0)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True, "smallheads": False} 

margins = [
    [100]*8,
    [400]*8,
    [10]*8,
    [1000]*8,
]

for fuzzy in [0]:
    for hingeLoss in [2]:
        for idxs, margin in enumerate(margins):
            torch.manual_seed(args.seed)
            np.random.seed(0)
            TASKS_NUM = 8
            net = models_jw.HashResNet18(100).to(device)
            train_loader = getters.get_dataset(args.dataset, args.period, args.batch_size, True, kwargs)
            test_loader = getters.get_dataset(args.dataset, 1, args.test_batch_size, False, kwargs)
            optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
            diffs = []
            int_id = 0
            for time in tqdm(range(TASKS_NUM*args.period)):
                task_idx = time // args.period
                if task_idx == int_id:
                    with open(f"exp_ebms/h{margin}_{fuzzy}_0_bh_{task_idx}", "wb") as fp:   
                        pickle.dump(diffs, fp)
                    torch.save(net.state_dict(), f"exp_ebms/h{margin}_{fuzzy}_0_bh_{task_idx}.pth")
                    int_id += 1

                X, y = train_loader.get_data()
                X, y = X.to(device), y.to(device)        
                z = prepare_z(task_idx, device, fuzzy)
                y_hat, _, _ = net(X, z)
                if True:
                    import torch
                    import torch.nn.functional as F
                    import random
                    def entropy_from_logits(logits):
                        probs = F.softmax(logits, dim=-1)
                        log_probs = torch.log(probs + 1e-9)  # Adding small epsilon to avoid log(0)
                        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
                        return entropy
                    z_wc = random.sample(list(range(task_idx, 10)), 1)
                    y_hat_wc, _, _ = net(X, prepare_z(z_wc, device, 0))
                optimizer.zero_grad()
                loss = hinge_loss(y_hat, y, margin[task_idx], type=hingeLoss, task_num=task_idx) - (margin[task_idx]/10)* entropy_from_logits(y_hat_wc[:,task_idx*10:(task_idx+1)*10])
                loss.backward()
                optimizer.step()
                diffs.append(torch.mean(y_hat.max(1).values - y_hat.min(1).values).item())

            # with open(f"ebms/{hingeLoss}_{idxs}_{fuzzy}_0_bh", "wb") as fp:   
            #     pickle.dump(diffs, fp)
            # torch.save(net.state_dict(), f"ebms/{hingeLoss}_{idxs}_{fuzzy}_0_bh.pth")