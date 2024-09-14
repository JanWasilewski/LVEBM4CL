import torch

def hinge_loss_old(y_hat, gt, margin=1.0):
  proper_pred = torch.gather(y_hat, 1, gt.view(-1, 1)).squeeze()  
  mask = torch.ones(y_hat.size(), dtype=torch.bool)
  rows = torch.arange(y_hat.size(0))
  mask[rows, gt] = False
  y_hat_wrongs = y_hat[mask].reshape(y_hat.size(0), y_hat.size(1) - 1)
  loss = torch.clamp(proper_pred.unsqueeze(1).repeat(1,3) - y_hat_wrongs + 1, min=0).sum()
  return loss


def hinge_loss(y_hat, gt, margin=1.0):
    # Get the predicted values for the correct class (ground truth)
    proper_pred = torch.gather(y_hat, 1, gt.view(-1, 1)).squeeze()  
    
    # Create a mask that excludes the correct class for each prediction
    mask = torch.ones_like(y_hat, dtype=torch.bool)
    rows = torch.arange(y_hat.size(0))
    mask[rows, gt] = False
    
    # Extract predictions for the wrong classes
    y_hat_wrongs = y_hat[mask].reshape(y_hat.size(0), y_hat.size(1) - 1)
    
    # Compute the hinge loss for all wrong predictions
    loss = torch.clamp(proper_pred.unsqueeze(1) - y_hat_wrongs + margin, min=0).sum(dim=1).mean()
    
    return loss


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