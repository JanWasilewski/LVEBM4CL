import torch

def hinge_loss_old(y_hat, gt, margin=1.0):
  proper_pred = torch.gather(y_hat, 1, gt.view(-1, 1)).squeeze()  
  mask = torch.ones(y_hat.size(), dtype=torch.bool)
  rows = torch.arange(y_hat.size(0))
  mask[rows, gt] = False
  y_hat_wrongs = y_hat[mask].reshape(y_hat.size(0), y_hat.size(1) - 1)
  loss = torch.clamp(proper_pred.unsqueeze(1).repeat(1,3) - y_hat_wrongs + 1, min=0).sum()
  return loss


def hinge_loss(y_hat, gt, margin=1.0, type=0, task_num=None):
    # Get the predicted values for the correct class (ground truth)
    proper_pred = torch.gather(y_hat, 1, gt.view(-1, 1)).squeeze()  
    # Create a mask that excludes the correct class for each prediction
    mask = torch.ones_like(y_hat, dtype=torch.bool)
    rows = torch.arange(y_hat.size(0))
    mask[rows, gt] = False
    
    # Extract predictions for the wrong classes
    y_hat_wrongs = y_hat[mask].reshape(y_hat.size(0), y_hat.size(1) - 1)
    # if type == 0: # the average logit
    #   y_hat_wrongs = y_hat_wrongs / y_hat_wrongs.shape[0]
    if type == 1: # the worst logit
      if task_num is not None:
        y_hat_wrongs = y_hat_wrongs[:,10*task_num:10*task_num+9]
      y_hat_wrongs = y_hat_wrongs.min(1).values.unsqueeze(1)
      
    if type == 2:  # random logit
      if task_num is not None:
        y_hat_wrongs = y_hat_wrongs[:,10*task_num:10*task_num+9]
      y_hat_wrongs = y_hat_wrongs[torch.arange(y_hat_wrongs.size(0)), torch.randint(0, y_hat_wrongs.size(1), (y_hat_wrongs.size(0),))].unsqueeze(1)
    # Compute the hinge loss for all wrong predictions
    loss = torch.clamp(proper_pred.unsqueeze(1) - y_hat_wrongs + margin, min=0).sum(dim=1).mean()
    
    return loss

def prepare_z(task_idx, device, fuzzy, noise=0.05):
    z = torch.zeros(10, device=device)
    z[task_idx] = 1
    if fuzzy:
        remaining_indices = [i for i in range(10) if i != task_idx]
        random_values = torch.rand(len(remaining_indices), device=device) * noise
        z[remaining_indices] = random_values
        z /= z.sum()
    return z

def square_square_loss(y_hat, gt, margin=1.0):
  y_hat = y_hat
  N = y_hat[0].shape[0]
  proper_pred = torch.gather(y_hat, 1, gt.view(-1, 1)).squeeze()
  mean = (torch.sum(y_hat, dim=1) - proper_pred)/(N-1)
  loss = (proper_pred**2 + (mean - margin*torch.ones_like(mean))**2).mean()
  return loss

def get_dataloader(dataset, task_num):
  if dataset == "CIFAR":
    a=2