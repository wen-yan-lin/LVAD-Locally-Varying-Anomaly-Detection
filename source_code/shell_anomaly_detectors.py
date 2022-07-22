import timeit
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import tensorflow as tf
#from dagmm import DAGMM
#from DeepUnsupAD import DistributionCluster
#from DeepUnsupAD.de2e import DE2E


def est_cummilative_dist(distance, multiples = 10):
    """ estimate the threshold for each percentile"""
    steps = np.array(range(100 * multiples)) / multiples
    cummilative = np.percentile(distance, steps)
    return cummilative

def percentile_from_score(cummulative_dist, scores):
    a = np.abs(cummulative_dist - scores)
    prob = np.argmin(a, axis=1) / cummulative_dist.size
    return prob




class ClusterLearner():

    def __init__(self, 
                num_clus = 5, 
                cluster_centers = None,
                ):
        self.num_clus = num_clus
        self.cluster_centers = cluster_centers
        self.stats = None

    def fit(self, data):
        kmeans = KMeans(n_clusters=self.num_clus, random_state=0).fit(data)
        self.cluster_centers = kmeans.cluster_centers_        
        
    def score_samples(self, test_data):        
        d = pairwise_distances(test_data, self.cluster_centers)
        return -np.min(d, axis=1)
                 

class LVAD():

    def __init__(self, 
                max_num_clus=300, 
                cluster_centers=None,
                display_time = False
                ):
        self.max_num_clus = max_num_clus
        self.cluster_centers = cluster_centers
        self.stats = None
        self.display_time = display_time

        
    def fit(self, data_):

        num_clus = min(self.max_num_clus, data_.shape[0] // 10)

        start_time = timeit.default_timer()

        kmeans = KMeans(n_clusters=num_clus, random_state=0).fit(data_)
        self.cluster_centers = kmeans.cluster_centers_   
        c_stats = ClusterStats()
        c_stats.fit_feat_to_means(data_, self.cluster_centers)
        self.stats = c_stats

        stop_time = timeit.default_timer()
        if self.display_time:
            print('Training Time: ', stop_time - start_time, ' seconds')  


    
    def score_samples(self, data):
        start_time = timeit.default_timer()
        score = self.stats.inference(data)
        stop_time = timeit.default_timer()
        if self.display_time:
            print('Inference Time: ',  (stop_time - start_time) / data.shape[0], ' seconds per instance')  
        return score




class ClusterStats():
    """statistics associted with each set of clusters"""

    def __init__(self):
        pass
        
    def fit_feat_to_means(self, feat_in, means, min_pts_per_cluster=10):
        """ estimate percentile stats from class features"""
        dist_in = pairwise_distances(feat_in, means)**2 
        inds = np.argmin(dist_in, axis=1)

        num_clusters = means.shape[0]
        cum_in = []
        cum_out = []
        frac = []
        ratio = []

        for i in range(num_clusters):
            bad_cluster = False
            mask = inds ==i
            if sum(mask) > min_pts_per_cluster:
                cum_in_i = est_cummilative_dist(dist_in[mask, i])
                cum_out_i = est_cummilative_dist(dist_in[~mask,i])

                frac_sub = sum(mask) / mask.size
                
                clus_mean = np.mean(dist_in[mask, i])
                clus_std = np.std(dist_in[mask, i])
                all_in_clus = sum(dist_in[:,i] - clus_mean < 3 * clus_std)
                
                p_noty_div_p_y = mask.size / all_in_clus -1

                if all_in_clus < min_pts_per_cluster:
                    bad_cluster = True
            else: 
                bad_cluster = True

            if bad_cluster == False:
                cum_in.append(cum_in_i)
                cum_out.append(cum_out_i)
                frac.append(frac_sub)
                ratio.append(p_noty_div_p_y)
            else:
                cum_in.append(None)
                cum_out.append(None)
                ratio.append(np.nan)
                frac.append(0)
        
        self.mean = means
        self.ratio = ratio # ratio between fraction of instances outside of the cluster's generative distribution
                           # and fraction of instances inside the cluster's generative distribution
        self.cum_in = cum_in # cummulative distribution of in_cluster points
        self.cum_out = cum_out # cummulative distributioin of out_cluster points
        self.frac = frac # fraction of points in each cluster
        
    def inference(self, feat, epsilon = 0.000000001):
        """ infer the likelihood that feat belongs to this class"""
        dist = pairwise_distances(feat, self.mean)**2  

        prob_all = np.zeros([feat.shape[0], self.mean.shape[0]])
        for i in range(len(self.cum_in)):
            if self.cum_in[i] is None:
                continue

            p_x_y = percentile_from_score(self.cum_in[i], dist[:,i:i+1])
            p_x_not_y = percentile_from_score(self.cum_out[i], dist[:,i:i+1])

            ratio = self.ratio[i]
            prob  =  (epsilon + p_x_y) / (ratio * p_x_not_y + p_x_y + epsilon) 
            prob_all[:,i] = prob 

        ####################### 
        
        mag = np.sum(self.frac) 
        if mag < 0.000000001:
            mag = 0.000000001 

        prob_all = prob_all * self.frac / mag

        
        return np.sum(prob_all, axis=1)
        

