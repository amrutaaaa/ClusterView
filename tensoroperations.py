import torch
import numpy as np

class EarlyStopper:
    def __init__(self, patience=3,min_delta=0):
        self.patience = patience
        self.counter = 0
        self.min_delta = min_delta
        self.min_loss = np.inf

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def cart2z2(point):
  x,y=point[:,0], point[:,-1]
  return x+ 1j * y

def createTensors(data):
  tensorData = torch.tensor(data[0]).reshape(1,len(data[0]))
  for i in range(1, len(data)):
    newPoint = torch.tensor(data[i]).reshape(1,len(data[0]))
    tensorData = torch.cat([tensorData, newPoint])
  return tensorData

def tensorNormalize(D, range=(0,1)):
    minimum, maximum = range
    D_std = (D - torch.min(D)) / (torch.max(D) - torch.min(D))
    D_scaled = D_std * (maximum - minimum) + minimum
    return D_scaled

def tensorNormalizedStress(D_high, D_low):
    return torch.sum((D_high - D_low)**2) / torch.sum(D_high**2)

def tensorStress(original, projected):
  D_high = torch.cdist(original, original).flatten()
  D_low = torch.cdist(projected, projected).flatten()

  D_low = tensorNormalize(D_low)
  D_high = tensorNormalize(D_high)

  return tensorNormalizedStress(D_high, D_low)