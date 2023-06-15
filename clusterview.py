from ClusterView.clusters import *
from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
import networkx as nkx
from ClusterView.clusteroperations import *
from ClusterView.tensoroperations import *
from ClusterView.metrics import *
import torch

torch.set_printoptions(precision=20)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

class ClusterView:
    def __init__(self, data_, clustering="metis", initial="fdl", clusterNumber=2):
        scaler = MinMaxScaler()
        self.data_ = scaler.fit_transform(data_.astype('float32'))
        self.clustering = "metis"
        self.initial = initial
        self.clusterNumber = clusterNumber

    def gamma_mul(self, points):
        points = torch.transpose(points, 0, 1)
        shrinked_points =  torch.transpose((points*self.gamma), 0, 1)
        return shrinked_points
    
    def make_z(self, X):
        appendZeroes={}
        complexCoordinates = cart2z2(X)
        coordinatesByParts = [[[],[]] for i in range(self.clusterNumber)]
        for i in range(len(self.parts)):
          coordinatesByParts[self.parts[i]][0].append(torch.real(complexCoordinates[i]))
          coordinatesByParts[self.parts[i]][1].append(torch.imag(complexCoordinates[i]))

        maxLen = max([len(coordinatesByParts[i][0]) for i in range(self.clusterNumber)])


        #appending zeroes according to the maxLen
        for i in range(self.clusterNumber):
          toAppend = maxLen - len(coordinatesByParts[i][0])
          coordinatesByParts[i][0].extend([0.]*toAppend)
          coordinatesByParts[i][1].extend([0.]*toAppend)
          appendZeroes[i] = toAppend


        #making tensor z from coordinatesByParts
        firstCluster = torch.complex(torch.tensor(coordinatesByParts[0][0]), torch.tensor(coordinatesByParts[0][1]))
        secondCluster = torch.complex(torch.tensor(coordinatesByParts[1][0]), torch.tensor(coordinatesByParts[1][1]))

        z = torch.stack([firstCluster, secondCluster])

        for i in range(2,self.clusterNumber):
            cluster = torch.complex(torch.tensor(coordinatesByParts[i][0]), torch.tensor(coordinatesByParts[i][1]))
            z = torch.vstack([z, cluster])

        z = z.to(device)


        #subtracting cluster centre from all points of each cluster
        for i in range(self.clusterNumber):
          cx, cy = self.centerUsed[i][0], self.centerUsed[i][1]
          z[i]=z[i]-(cx + 1j*cy)

        return z, appendZeroes
    

    def forward(self, z, appendZeroes):

        #multiplying all points of one cluster to their corresponding gamma
        z=self.gamma_mul(z)

        newCoordinates=None

        for i in range(self.clusterNumber):
          cx, cy = self.centerUsed[i][0], self.centerUsed[i][1]

          #adding back the centre to all the points in a cluster
          z[i]=z[i]+(cx + 1j*cy)

          #uisng appendzeroes to use slicing to only take the points in the xluster that were originally in the cluster
          if newCoordinates==None:
            newCoordinates =torch.stack([torch.real(z[i]), torch.imag(z[i])], dim=1)[:-appendZeroes[i]] if appendZeroes[i]!=0 else torch.stack([torch.real(z[i]), torch.imag(z[i])], dim=1)
          else:
            catPoints = torch.stack([torch.real(z[i]), torch.imag(z[i])], dim=1)[:-appendZeroes[i]] if appendZeroes[i]!=0 else torch.stack([torch.real(z[i]), torch.imag(z[i])], dim=1)
            newCoordinates = torch.cat([newCoordinates, catPoints])

        return newCoordinates
    
    def gradientDescent(self, embeddings):
        self.gamma = torch.complex(torch.tensor([0.95]*self.clusterNumber),torch.tensor([0.]*self.clusterNumber))
        self.gamma = self.gamma.to(device)
        self.gamma.requires_grad_()

        X = embeddings

        z, appendZeroes = self.make_z(X)

        early_stopper = EarlyStopper(patience=20, min_delta=0)
        optimizer = torch.optim.Adam([self.gamma], lr=0.2)
        lossArray=[]
        n_iter = 1000
        for i in range(n_iter):
            # making predictions with forward pass
            Y_pred = self.forward(z, appendZeroes)
            # calculating the loss between original and predicted data points
            loss = tensorStress(self.originalByParts, Y_pred)
            # storing the calculated loss in a list
            lossArray.append(loss.item())
            optimizer.zero_grad()
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            loss.backward()
            # # updating the parameters after each iteration
            optimizer.step()

            #checking for early stoppping condition
            if early_stopper.early_stop(loss.item()):
                break

    def runClusterView(self):

        reducer = umap.UMAP()
        reducer.fit(self.data_)
        embedding = reducer.embedding_
        scaler = MinMaxScaler()
        scaledEmbedding =  scaler.fit_transform(embedding.astype('float32'))
        
        if self.clustering=="metis":
            graph = reducer.graph_
            ngraph = nkx.Graph(graph)
            self.parts = metisClustering(ngraph, self.clusterNumber)

        elif self.clustering=="dcanopy":
            self.parts, self.clusterNumber = densityCanopyClustering(self.data_)

        centre = calculateClusterCenter(self.data_, self.parts)
        initialCentres = calculateClusterCenter(scaledEmbedding, self.parts)

        if self.initial=="fdl":
            difference = posDifference(self.parts, centre, initialCentres)
        elif self.initial=="pca":
            difference = posDifference(self.parts, centre, initialCentres, "pca")

        translatedEmbedding = translateUsingDifference(difference, self.parts, scaledEmbedding)

        originalDataTensors = createTensors(self.data_)
        embeddingTensors = createTensors(translatedEmbedding)

        originalDataTensors = originalDataTensors.to(device)
        embeddingTensors = embeddingTensors.to(device)

        self.centerUsed = calculateClusterCenter(embeddingTensors, self.parts)
        tensorCenter=[]
        for i in range(len(self.parts)):
          tensorCenter.append(self.centerUsed[self.parts[i]])
        tensorCenter = createTensors(tensorCenter)
        tensorCenter = tensorCenter.to(device)

        self.centerUsed = {k: v.to(device=device) for k, v in self.centerUsed.items()}

        self.originalByParts = [[] for i in range(self.clusterNumber)]
        for i in range(len(self.parts)):
          self.originalByParts[self.parts[i]].append(self.data_[i])
        m = [len(self.originalByParts[x]) for x in range(self.clusterNumber)]
        a =[]
        for i in range(self.clusterNumber):
          for j in self.originalByParts[i]:
            a.append(j)
        self.originalByParts = createTensors(a)
        self.originalByParts = self.originalByParts.to(device)

        self.gradientDescent(embeddingTensors)

        self.final = shrinkrotateall(embeddingTensors, self.parts, self.gamma)

        return self.final
    
    def resultMetrics(self):
       resultStress = stress(self.data_, self.final.detach().numpy())
       trust, conti = TandC(self.data_, self.final.detach().numpy())
       resultComposite = compositeMetric([resultStress, trust, conti])

       return {"Stress": resultStress, "Trustworthiness":trust, "Continuity":conti, "Composite Metric": resultComposite}
