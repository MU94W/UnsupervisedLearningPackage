#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:56:06 2017

Unsupervised Learning Package.

@author: tpog
"""

import numpy as np
import numpy.random as random
from six.moves import xrange
import matplotlib.pyplot as plt
import os

def calDist(vec1, vec2):
    return sum([diff ** 2 for diff in (vec1 - vec2)]) ** 0.5

class USL(object):
    def __init__(self):
        """
        Object properties:
        dots: the number of dots (to be) clustered.
        dims: the number of features of a single dot.
        cluster_num: the number of centroids/medoids.
        data: the features of all the dots.(numeric)
        name: a list that contains the dots' names. It may be a path or the coordinate of a dot.
        clusterDotSets: a list, it contains '%d' % cluster_num arrays that save the numeric features of all the dots 
                        that belong to some one cluster-category.
        clusterIDSets: a list, it contains '%d' % cluster_num arrays that save the ID(i.e. the index in self.data or
                       self.name) of all the dots that belong to some one cluster-category.
        clusterID: a list, len(self.clusterID) == self.dots. If you want to know the 5th dot's cluster-category, you
                   can key down self.clusterID[5].
        """
        self.dots = 0
        self.dims = 0
        self.data = []
        self.name = []

    def readFile(self,dataFilePath):
        self.data = np.loadtxt(dataFilePath,dtype=float)
        (self.dots,self.dims) = self.data.shape
        return

    def readArray(self,arr):
        self.data = np.array(arr,dtype=float)
        (self.dots,self.dims) = self.data.shape
        return

    def readNameFeatFile(self, path):
        """
        Parse the file that contains dot names and corresponding features.
        Given file's content is assumed to be written like this:
        
        ---------------FILE CONTENT REGION---------------
        dots_num(int) features_num(int)
        dot_name(str)
        dot_features(num)(with comma seperated)
        dot_name(str)
        dot_features(num)(with comma seperated)
        ...
        ...
        dot_name(str)
        dot_features(num)(with comma seperated)
        ---------------FILE CONTENT REGION---------------

        Example file:

        ---------------FILE CONTENT REGION---------------
        4096 3
        (10,20)
        1.2,3,10,
        (0,-8)
        -1,350,2,
        ...
        ...
        ---------------FILE CONTENT REGION---------------

        Parameters
        ----------
        path: str
            the path to get your file

        """
        handle = open(path, 'r')
        data = handle.read().split('\n')
        handle.close()
        dotsanddims = data[0].split(' ')
        self.dots = int(dotsanddims[0])
        self.dims = int(dotsanddims[1])
        data = data[1:]
        # get feat
        feats = np.empty((self.dots,self.dims),dtype=float)
        for i in range(self.dots):
            featstr = data[2*i+1].split(',')
            featnum = [float(featstr[k]) for k in range(self.dims)]
            feats[i] = featnum
        self.data = feats
        # get name
        names = []
        for i in range(self.dots):
            names.append(data[2*i])
        self.name = names
        return

    def zscore(self):
        """
        For the i-th feature-dim:
        mean(i) := the arithmetic mean of all dots' i-th feature.
        std(i)  := the standard variation of all dots' i-th feature.

        Converting Formula:
            new_feat(i) = (old_feat(i) - mean(i)) / std(i)
        """
        mean = self.data.mean(axis=0)
        self.mean = mean
        std = self.data.std(axis=0)
        self.std = std
        for index in range(self.dims):
            assert std[index] > 1E-20, "ERROR! The %d-th feature's std is too small!" % index
        self.data = (self.data - mean) / std



class Cluster(USL):
    def __init__(self):
        """
        Object properties:
        dots: the number of dots (to be) clustered.
        dims: the number of features of a single dot.
        cluster_num: the number of centroids/medoids.
        data: the features of all the dots.(numeric)
        name: a list that contains the dots' names. It may be a path or the coordinate of a dot.
        clusterDotSets: a list, it contains '%d' % cluster_num arrays that save the numeric features of all the dots 
                        that belong to some one cluster-category.
        clusterIDSets: a list, it contains '%d' % cluster_num arrays that save the ID(i.e. the index in self.data or
                       self.name) of all the dots that belong to some one cluster-category.
        clusterID: a list, len(self.clusterID) == self.dots. If you want to know the 5th dot's cluster-category, you
                   can key down self.clusterID[5].
        """
        self.cluster_num = 0
        self.clusterDotSets = []
        self.clusterIDSets = []
        self.clusterID = []
        super(Cluster,self).__init__()
 
    def cluster(self):
        raise NotImplementedError

    def saveResult(self,dir_path):
        """
        Save the cluster procedure's result.
        You will get a dir(dir_path) which contains '%d' % cluster_num files that contains corresponding dot names
        belonging to some cluster-category.
        """
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for index in xrange(self.cluster_num):
            with open(os.path.join(dir_path,'%d.txt'%index),'w+') as f:
                for nameindex in self.clusterIDSets[index]:
                    f.write(self.name[nameindex] + '\n')

    def showData(self):
        """
        Show the imported data on a 2D canvas.
        That implies only the first two features are drawn.
        """
        assert len(self.data) != 0, "The data haven't been loaded."
        plt.figure()
        data_t = self.data.transpose()
        plt.scatter(data_t[0], data_t[1])
        plt.legend()
        plt.show()
        
    def showClusteredData(self):
        """
        Show the clustered data on a 2D canvas.
        That implies only the first two features are drawn.
        Except that, now this method just can plot the first five feature-dims.
        """
        assert len(self.clusterDotSets) != 0, "Please cluster the data ahead."
        plt.figure()
        colorSet = ['r','g','b','y','k']
        for clt_index in xrange(self.cluster_num):
            data_t = self.clusterDotSets[clt_index].transpose()
            plt.scatter(data_t[0], data_t[1], c=colorSet[clt_index])
        data_t = self.centroids.transpose()
        plt.scatter(data_t[0], data_t[1], s=10, c='k', marker='*')
        plt.legend()
        plt.show()

class DenPeak(Cluster):
    def __init__(self):
        self.dc = 0
        self.distTable = []
        self.rhoTable = []
        self.deltaTable = []
        self.parentTable = []
        self.noiseDotSet = []
        self.noiseIDSet = []
        super(DenPeak, self).__init__()

    def cluster(self,dc=0.07,cluster_num=10,rho_min=100,noise_detec=False):
        self.calProp(dc)
        self.assigncluster(cluster_num,rho_min,noise_detec)
        
    def calProp(self,dc=0.07):
        self.dc = dc
        self.getDistTable()
        self.getRhoTable()
        self.getDeltaTable()
        
    def getDistTable(self):
        self.distTable = np.empty([self.dots, self.dots], dtype=float)
        for i in xrange(self.dots):
            for j in xrange(i,self.dots):
                self.distTable[i][j] = calDist(self.data[i], self.data[j])
                self.distTable[j][i] = self.distTable[i][j]
        return
    
    def getRhoTable(self):
        self.rhoTable = np.empty([self.dots], dtype = float)
        def calRho(vec):
            return sum([1 if x < self.dc else 0 for x in vec])
        for i in xrange(self.dots):
            self.rhoTable[i] = calRho(self.distTable[i])
        neighborTable = []
        for i in xrange(self.dots):
            rho = 0
            d_v = self.distTable[i]
            neighbors = []
            for j in xrange(self.dots):
                if d_v[j] < self.dc:
                    rho += 1
                    neighbors.append(j)
            neighborTable.append(neighbors)
        self.neighborTable = neighborTable
        return
    
    def getDeltaTable(self):
        self.deltaTable = np.empty([self.dots], dtype = float)
        self.parentTable = np.full([self.dots],-1, dtype = int)
        for i in xrange(self.dots):
            delta = 1E9
            parent = -1
            for j in xrange(self.dots):
                if i == j:
                    continue
                if self.rhoTable[j] > self.rhoTable[i]:
                    if self.distTable[i][j] < delta:
                        delta = self.distTable[i][j]
                        parent = j
            if parent != -1:
                self.deltaTable[i] = delta
            else:
                self.deltaTable[i] = max(self.distTable[i])
            self.parentTable[i] = parent
        return
        
    def assigncluster(self,cluster_num,rho_min,noise_detec=True):
        self.cluster_num = cluster_num
        # find medoids
        indexedDeltaTable = np.column_stack((self.deltaTable,np.arange(0,self.dots)))
        sortedDeltaTable = sorted(indexedDeltaTable,key=lambda x: x[0],reverse=True)
        medoidsID = []
        found = 0
        def isRealMedoid(n):
            for i in medoidsID:
                if self.distTable[n][i] < self.dc:
                    self.parentTable[n] = i
                    return False
            return True
        for i in xrange(self.dots):
            if found == cluster_num:
                break
            dot_ID = sortedDeltaTable[i][1]
            dot_rho = self.rhoTable[dot_ID]
            if dot_rho > rho_min and isRealMedoid(dot_ID):
                medoidsID.append(dot_ID)
                found += 1
        self.medoidsID = np.array(medoidsID,dtype=int)
        # end
        # assign cluster medoids
        clusterID = np.full(self.dots,-1,dtype=int)
        clt_tmp = 0
        for i in medoidsID:
            clusterID[i] = clt_tmp
            clt_tmp += 1
        # end
        # assign remaining dots
        def assignClusterID(n):
            if n == -1:
                return -1
            my_clt_id = clusterID[n]
            if my_clt_id >= 0:
                return my_clt_id
            else:
                parent_clt_ID = assignClusterID(self.parentTable[n])
                clusterID[n] = parent_clt_ID
                return parent_clt_ID
        for i in xrange(self.dots):
            assignClusterID(i)
        # end
        # get borders and find noise dots
        if noise_detec:
            def isBorderDot(clt_id,vec):
                for i in vec:
                    neighbor_clt_ID = clusterID[i]
                    if neighbor_clt_ID != -1 and neighbor_clt_ID != clt_id:
                        return True
                return False
            clusterRhoB = np.zeros(self.cluster_num)
            for i in xrange(self.dots):
                if isBorderDot(clusterID[i], self.neighborTable[i]):
                    rho = self.rhoTable[i]
                    if rho > clusterRhoB[clusterID[i]]:
                        clusterRhoB[clusterID[i]] = rho
            for i in xrange(self.dots):
                if clusterID[i] == -1:
                    continue
                if self.rhoTable[i] < clusterRhoB[clusterID[i]]:
                    clusterID[i] = -1
        # end
        # split to each cluster
        clusterDotSets = []
        clusterIDSets = []
        noiseDotSet = []
        noiseIDSet = []
        for clt_index in xrange(cluster_num):
            clusterDotSets.append([])
            clusterIDSets.append([])
        for i in xrange(self.dots):
            cluster_ID = clusterID[i]
            dot_data = self.data[i]
            if cluster_ID != -1:
                clusterDotSets[cluster_ID].append(dot_data)
                clusterIDSets[cluster_ID].append(i)
            else:
                noiseDotSet.append(dot_data)
                noiseIDSet.append(i)
        for clt_index in xrange(cluster_num):
            clusterDotSets[clt_index] = np.array(clusterDotSets[clt_index])
            clusterIDSets[clt_index] = np.array(clusterIDSets[clt_index])
        self.clusterDotSets = clusterDotSets
        self.clusterIDSets = clusterIDSets
        self.noiseDotSet = np.array(noiseDotSet)
        self.noiseIDSet = np.array(noiseIDSet)
        self.clusterID = clusterID
        # end
    
    def showDecisionGraph(self):
        plt.figure()
        plt.scatter(self.rhoTable, self.deltaTable)
        plt.xlabel("rho")
        plt.ylabel("delta")
        plt.title("Decision Graph")
        plt.legend()
        plt.show()
        
class Kmeans(Cluster):
    def __init__(self):
        super(Kmeans, self).__init__()

    def cluster(self,cluster_num,iters=10):
        self.cluster_num = cluster_num
        # init
        centroids = []
        tmp_id = []
        clt =  0
        data = self.data
        while clt < cluster_num:
            randid = random.randint(0,self.dots)
            if randid in tmp_id:
                continue
            else:
                tmp_id.append(randid)
                clt += 1
        for i in xrange(cluster_num):
            centroids.append(data[tmp_id[i]])
        # end
        # begin iteration
        clusterID = np.empty(self.dots,dtype=int)
        def getClusterID(n):
            now_dist = 1E9
            now_id = -1
            for i in xrange(cluster_num):
                tmp_dist = calDist(data[n],centroids[i])
                if tmp_dist < now_dist:
                    now_dist = tmp_dist
                    now_id = i
            return now_id
        def updateCentroid(n):
            tmp_sum = np.zeros(self.dims,dtype=float)
            tmp_cnt = 0
            for i in xrange(self.dots):
                if clusterID[i] == n:
                    tmp_sum += data[i]
                    tmp_cnt += 1
            new_cent = tmp_sum / tmp_cnt
            return new_cent
            
        for its in xrange(iters):
            # assgin cluster_ID
            for i in xrange(self.dots):
            	clusterID[i] = getClusterID(i)
            # end
            # update centroids
            for i in xrange(cluster_num):
                centroids[i] = updateCentroid(i)
            # end
        self.centroids = np.array(centroids)
        # end
        # split to each cluster
        clusterDotSets = []
        clusterIDSets = []
        for clt_index in xrange(cluster_num):
            clusterDotSets.append([])
            clusterIDSets.append([])
        for i in xrange(self.dots):
            cluster_ID = clusterID[i]
            dot_data = self.data[i]
            clusterDotSets[cluster_ID].append(dot_data)
            clusterIDSets[cluster_ID].append(i)
        for clt_index in xrange(cluster_num):
            clusterDotSets[clt_index] = np.array(clusterDotSets[clt_index])
            clusterIDSets[clt_index] = np.array(clusterIDSets[clt_index])
        self.clusterDotSets = clusterDotSets
        self.clusterIDSets = clusterIDSets
        self.clusterID = clusterID
        # end

    def sortByDist(self):
        assert self.cluster_num > 0, 'You should do CLUSTER method first!'
        dt = np.dtype([('ID',np.int),('dist',np.float32)])
        combinedSets = []
        # cal the necessary dist info, and combine it with the corresponding ID.
        for clt_index in xrange(self.cluster_num):
            centroid = self.centroids[clt_index]
            combinedSet = np.empty(shape=self.clusterIDSets[clt_index].shape,dtype=dt)
            cnt = 0
            for dot,ID in zip(self.clusterDotSets[clt_index],self.clusterIDSets[clt_index]):
                dist = calDist(centroid,dot)
                combinedSet[cnt] = (ID,dist)
                cnt += 1
            combinedSets.append(combinedSet)
        # sort by dist
        for clt_index in xrange(self.cluster_num):
            combinedSets[clt_index] = np.sort(combinedSets[clt_index],order='dist')
        self.sortCombinedSets = combinedSets

    def saveSortedResult(self,dir_path):
        """
        Save the sortByDist procedure's result.
        You will get a dir(dir_path) which contains '%d' % cluster_num files that contains corresponding dot names
        belonging to some cluster-category sorted by dist.
        """
        assert hasattr(self,'sortCombinedSets'), 'You should do SORT_BY_DIST method first!'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for index in xrange(self.cluster_num):
            with open(os.path.join(dir_path,'sorted_%d.txt'%index),'w+') as f:
                for item in self.sortCombinedSets[index]:
                    f.write(self.name[item[0]] + '\n')


from math import log,pi
class Xmeans(Kmeans):
    def __init__(self):
        super(Xmeans,self).__init__()

    def distort(self,coeflist):
        for i in range(len(coeflist)):
            self.data[:,i] *= coeflist[i]

    """
    def calBIC(self,dot_set,R,K):
        Rn = dot_set.shape[0]
        M = dot_set.shape[1]
        var = dot_set.var(axis=0).sum() * Rn / (Rn-K)
        loglikelihood = - Rn/2*log(2*pi) - Rn*M/2*log(var) - (Rn-K)/2 + Rn*log(Rn) - Rn*log(R)
        pj = K*(M+1) # pj = (K-1) + M*K + 1
        BIC = loglikelihood - pj/2 * log(Rn)
        return BIC
    """
    def calBIC(self,sets,centroids,K):
        if K == 1:
            dots = sets.shape[0]
            dims = sets.shape[1]
            var = sets.var(axis=0).sum() * dots/(dots-K)
            loglikelihood = - dots/2*log(2*pi) - dots*dims/2*log(var) - (dots-K)/2
        elif K == 2:
            dots1 = sets[0].shape[0]
            dots2 = sets[1].shape[0]
            dots = dots1 + dots2
            dims = sets[1].shape[1]
            var = (sets[0].var(axis=0).sum()*dots1 + sets[1].var(axis=0).sum()*dots2) / (dots-K)
            loglikelihood = dots1*log(dots1/dots) + dots2*log(dots2/dots) - dots/2*log(2*pi) - dots*dims/2*log(var) - (dots-K)/2
        else:
            assert False, 'K := {1,2}, this K = %d.' % K 
        freeparameters = K*(dims+1) # = (K-1) + dims*K + 1
        logpostprob = loglikelihood - freeparameters/2*log(dots)
        return logpostprob

    def splitCalBIC(self,dot_set,centroid,iters=10):
        localhandle = Kmeans()
        localhandle.readArray(dot_set)
        localhandle.cluster(2,iters)
        """
        R = dot_set.shape[0]
        M = dot_set.shape[1]
        K1BIC = self.calBIC(dot_set,R,1)
        K2BIC = self.calBIC(localhandle.clusterDotSets[0],R,2) + self.calBIC(localhandle.clusterDotSets[1],R,2)
        """
        K1BIC = self.calBIC(dot_set,centroid,1)
        K2BIC = self.calBIC(localhandle.clusterDotSets,localhandle.centroids,2)
        return K1BIC,K2BIC

    def shouldIncreaseCents(self,iters=10):
        cluster_num = self.centroids.shape[0]
        for i in xrange(cluster_num):
            K1BIC,K2BIC = self.splitCalBIC(self.clusterDotSets[i],self.centroids[i],iters)
            print('BIC(k=1) = %f' % K1BIC)
            print('BIC(k=2) = %f' % K2BIC)
            if K1BIC < K2BIC:
                print('Increase!')
                return True
        return False

    def cluster(self,init_cluster_num=2,iters=10):
        super(Xmeans,self).cluster(init_cluster_num,iters)
        should = self.shouldIncreaseCents(iters)
        cluster_num = init_cluster_num
        while should:
            cluster_num += 1
            print('Now cluster num = %d' % cluster_num)
            super(Xmeans,self).cluster(cluster_num,iters)
            should = self.shouldIncreaseCents(iters)
        print('Final cluster num = %d' % cluster_num)
        return cluster_num

class PCA(USL): 
    def __init__(self):
        """
        You should do:
        1st: preProcess()
        2nd: selectEig(proportion)
        3rd: posProcess(path), where 'path' is the place saving the transed data
        
        Object properties:
        eigVals:
        eigVecs:
        eigValProps:
        selEigVals:
        selEigVecs:
        """
        super(PCA,self).__init__()

    def preProcess(self):
        """
        Add the following properties:
        eigVals: 
        eigVecs:
        eigValProps:
        """
        covMat = np.mat(np.cov(self.data,rowvar=False))
        eigVals,eigVecs = np.linalg.eig(covMat)
        eigValIndx = np.argsort(eigVals)[::-1]
        self.eigVals = eigVals[eigValIndx]
        self.eigVecs = eigVecs[:,eigValIndx]
        propsum = np.sum(self.eigVals)
        self.eigValProps = self.eigVals / propsum

    def selectEig(self,proportion):
        tmppropsum = 0
        cnt = 0
        for eigvp in self.eigValProps:
            if tmppropsum < proportion:
                tmppropsum += eigvp
                cnt += 1
        self.selEigVals = self.eigVals[:cnt]
        self.selEigVecs = self.eigVecs[:,:cnt]

    def posProcess(self,path):
        self.transData = self.data * self.selEigVecs
        dims = self.selEigVals.shape[0]
        with open(path,'w+') as f:
            content = '%d %d\n' % (self.dots,dims)
            for i in range(self.dots):
                content += self.name[i] + '\n'
                for j in range(dims):
                    content += '%f' % self.transData[i,j] + ','
                content += '\n'
            f.write(content)
                


 
