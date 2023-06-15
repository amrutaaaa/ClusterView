import numpy as np
import forcelayout as fl
from sklearn.decomposition import PCA
import torch

def calculateClusterCenter(data, parts):
  centre={}
  for i in range(len(parts)):
    if parts[i] in centre:
      centre[parts[i]]+=data[i]
    else:
      centre[parts[i]]=data[i]


  for j in centre:
      centre[j]/=parts.count(j)

  return centre

def selectCluster(parts, points, cluster):
  '''
    returns all the points contained in a particular cluster number

    parts: an array containing the cluster number of every point in the data
    points: an array containing all the points
    cluster: an integer which tells which cluster number is to be selected
  '''
  
  clusterPoints=[]
  for i in range(len(parts)):
    if parts[i]==cluster:
      clusterPoints.append(list(points[i]))
  return np.array(clusterPoints)

def translate(cluster,distances):
  return cluster+distances

def changeCluster(parts, coordinates, projected, cluster):
  count=0
  newProjected=[]
  for i in range(len(parts)):
    if parts[i]==cluster:
      newProjected.append(coordinates[count])
      count+=1
    else:
      newProjected.append(projected[i])
  return newProjected

def forceDirectedCentres(originalCentres, parts):
  clusterCentres = np.array(originalCentres[0])
  for i in range(1,len(set(parts))):
    clusterCentres=np.vstack((clusterCentres, originalCentres[i]))
  layout = fl.draw_spring_layout(dataset=clusterCentres, algorithm=fl.SpringForce)
  forceDirectedPositions = layout.get_positions()
  return forceDirectedPositions

def PCACentres(originalCentres, parts):
  clusterCentres = np.array(originalCentres[0])
  for i in range(1,len(set(parts))):
    clusterCentres=np.vstack((clusterCentres, originalCentres[i]))
  pcaCoords = PCA(n_components=2).fit_transform(clusterCentres)
  return pcaCoords

def posDifference(parts, highDimensionCentres, lowDimensionCentres, initialization="fdl"):
  if initialization=="fdl":
    centerRepresentation = forceDirectedCentres(highDimensionCentres, parts)
  elif initialization =="pca":
    centerRepresentation = PCACentres(highDimensionCentres, parts)

  originalLowCentres = np.array(lowDimensionCentres[0])
  for i in range(1,len(set(parts))):
    originalLowCentres=np.vstack((originalLowCentres, lowDimensionCentres[i]))

  return centerRepresentation - originalLowCentres

def translateUsingDifference(difference, parts, embedding):
  newEmbedding = []
  for i in range(len(parts)):
    translated = translate(embedding[i], difference[parts[i]])
    newEmbedding.append(translated)
  return np.array(newEmbedding)

def cart2z(point):
  x,y=point
  return x+ 1j * y

def z2cart(z):
  return torch.tensor([torch.real(z), torch.imag(z)])

def shrinkrotate(point, centre, gamma):
  cx,cy=centre
  z=cart2z(point)
  z=z-(cx + 1j*cy)
  z=gamma*z
  z=z+(cx + 1j*cy)
  return z2cart(z)

def shrinkrotateall(embedding, parts, gamma):

  '''
  translatedEmbedding : takes in a 2d array - 2dimensionals points
  parts : parts[i] signifies the cluster number ith point belongs to
  gamma : len(gamma) = #number of clusters, gamma[i] signfies the shrinking factor of the ith cluster,

  '''

  #finding the cluster centres from the Embedding (centroids)
  clusterCentres=calculateClusterCenter(embedding,parts)

  # #if gamma is uninitialized, default it to 0.95 for all clusters
  # if(gamma==None):
  #   gamma = [0.95]*len(clusterCentres)

  # #if theta is uninitialized, default it to 0 for all clusters
  # if(theta==None):
  #   theta = [0]*len(clusterCentres)



  newCoordinates=[]
  #iterating through all the points
  for i in range(len(parts)):
    newCoordinates.append(shrinkrotate(embedding[i], clusterCentres[parts[i]], gamma[parts[i]]))

  tensorCoordinates = newCoordinates[0].clone().detach().requires_grad_(True).reshape(1, len(newCoordinates[0]))

  for i in range(1, len(newCoordinates)):
    newPoint = newCoordinates[i].clone().detach().requires_grad_(True).reshape(1, len(newCoordinates[0]))
    tensorCoordinates = torch.cat([tensorCoordinates, newPoint])


  # newCoordinates = np.array(newCoordinates)
  return tensorCoordinates