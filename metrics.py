from scipy import spatial
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyDRMetrics.coranking_matrix import *
from sklearn.metrics import pairwise_distances
import pandas as pd


def normalize(D):
    scaler = MinMaxScaler()
    D = scaler.fit_transform(D.reshape((-1, 1)))
    D = D.squeeze()
    return D

def normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)

def stress(originalData, projectedData):
  D_low = spatial.distance.pdist(projectedData, 'euclidean')
  D_high = spatial.distance.pdist(originalData, 'euclidean')
  
  D_low = normalize(D_low)
  D_high = normalize(D_high)
  
  return normalized_stress(D_high, D_low)

def TandC(X,Z):
  df = pd.DataFrame(X, index=None)
  D = pd.DataFrame(pairwise_distances(df.values)).values
  dfz = pd.DataFrame(Z, index=None)
  Dz = pd.DataFrame(pairwise_distances(dfz.values)).values

  R = ranking_matrix(D)
  Rz = ranking_matrix(Dz)
  Q = coranking_matrix(R, Rz)

  Q = Q[1:,1:]
  m = len(Q)

  Qs = Q[7:,:7]
  W = np.arange(Qs.shape[0]).reshape(-1, 1) # a column vector of weights. weight = rank error = actual_rank - k
  T = 1-np.sum(Qs * W)/(7+1)/m/(m-1-7)  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
  Qs = Q[:7,7:]
  W = np.arange(Qs.shape[1]).reshape(1, -1) # a row vector of weights. weight = rank error = actual_rank - k
  C = 1-np.sum(Qs * W)/(7+1)/m/(m-1-7)  # 1 - normalized hard-k-extrusions. upper-right region 

  return (T,C)

def compositeMetric(metrics):
  '''
    returns the value of composite metric mu

    metrics: an array which contains stress value at 0th index, trustworthiness at 1st index and continuity at 2nd index
  '''
  return ((1-metrics[0])+metrics[1]+metrics[2])/3