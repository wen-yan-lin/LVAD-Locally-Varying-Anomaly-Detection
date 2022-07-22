""" <Local Varying Anomaly Detection>
    Copyright (C) <2022>  <Wen-Yan Lin>
    daniellin@smu.edu.sg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""

import numpy as np
import copy
from shell_anomaly_detectors import LVAD
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

def normIt(data, m=None):
    nData = data.copy()
    if m is None:
        m = np.mean(nData, axis =0, keepdims=True)
    nData = nData - m
    nData = nData / np.linalg.norm(nData, axis =1, keepdims=True)
    
    return nData, m




class NoNormalization():
    def __init__(self):
        self.ref = None
    
    def fit(self, data):
        pass

    def transform(self, data):
        return copy.copy(data)


class InstanceNormalization():
    def __init__(self):
        self.ref = None
    
    def fit(self, data):
        pass

    def transform(self, data):
        m = np.mean(data, axis=1, keepdims=True)
        data_ = data - m
        data_ = data_ / np.linalg.norm(data_, axis=1, keepdims=True) 
        return data_

class NaiveNormalization():
    def __init__(self):
        self.ref = None
    
    def fit(self, data):
        self.ref = np.mean(data, axis=0)


    def transform(self, data):
        data_, _ = normIt(data, self.ref)
        return data_

class ErgoNormalization():
    def __init__(self, ref=None):
        self.ref = None
    
    def fit(self, data):
        self.ref = np.mean(data)
    
    def transform(self, data):
        data_, _ = normIt(data, self.ref)
        return data_

class PreTrainedNormalization():
    def __init__(self, ref):
        self.ref = ref
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        data_, _ = normIt(data, self.ref)
        return data_


class NormalizedAnomalyDetector():
    
    def __init__(self,
                clf = LVAD(max_num_clus = 1000),
                norm = InstanceNormalization()):
        self.clf = clf
        self.norm = norm
        
    def fit(self, data):
        self.norm.fit(data)
        data_ = self.norm.transform(data)
        self.clf.fit(data_)
    
    def score_samples(self, data):       
        data_ = self.norm.transform(data)
        return self.clf.score_samples(data_)


class NormalizedPCALearner():

    def __init__(self, 
                num_dim = 3, 
                kde_bandwidth = 0.2,
                norm = InstanceNormalization()
                ):
        self.num_dim = num_dim
        self.kde_bandwith = kde_bandwidth
        self.norm = norm  
        self.kde = None
      
    def fit(self, train_data):
        self.norm.fit(train_data)
        data_ = self.norm.transform(train_data)

        pca = PCA(n_components=self.num_dim)
        pca.fit(data_)
        data_ = pca.transform(data_)
        kde = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwith).fit(data_)

        self.pca = pca
        self.kde = kde
    
    def score_samples(self, test_data):
        test_data_ = self.norm.transform(test_data)
        test_data_ = self.pca.transform(test_data_)
        return  self.kde.score_samples(test_data_)