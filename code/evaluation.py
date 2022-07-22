import numpy as np
import random
import matplotlib.pyplot as plt

from normalization import InstanceNormalization, ErgoNormalization, NaiveNormalization, NoNormalization, PreTrainedNormalization
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances

def quick_scoredisplay_scores(clf, data, gt):
    clf.fit(data)
    score = clf.score_samples(data)

    num = sum(gt==0)
    print('auroc:', roc_auc_score(gt, score))

    plt.hist(score[gt==0][:num], alpha=0.5)
    plt.hist(score[gt==1][:num], alpha=0.5)


def build_eval_set(x_train, y_train, ind, p_anon):
    x_in = x_train[y_train==ind]
    x_out = x_train[y_train!=ind]
    random.shuffle(x_out)

    num_out = int(p_anon / 100 * x_in.shape[0])
    data = np.concatenate([x_in, x_out[:num_out]], axis=0)
    gt = np.zeros(data.shape[0], dtype=int)
    gt[:data.shape[0] -num_out] = 1
    
    return data, gt



class AnonEvaluationStatistics():
    def __init__(self, 
                 percentiles =[0.1, 1, 10, 20, 30], 
                 name = 'unnamed'):
        self.percentiles = percentiles
        self.name = name
        self.auroc = None
        self.auprc = None
        self.mean_auroc = None
        self.mean_auprc = None
    
    
    def eval(self, x_train, y_train, clf, print_summary=True):
        num_class = np.max(y_train) + 1
        
        auroc_scores = np.zeros([num_class, len(self.percentiles)])
        auprc_scores = np.zeros([num_class, len(self.percentiles)])

        for class_num in range(num_class):
            for anon_ind, p_anon in enumerate(self.percentiles):
                data, gt = build_eval_set(x_train, y_train, class_num, p_anon)
                
                clf.fit(data)
                score = clf.score_samples(data)
                auroc = roc_auc_score(gt, score)
                auprc = average_precision_score(gt, score)

                
                auroc_scores[class_num, anon_ind] = auroc
                auprc_scores[class_num, anon_ind] = auprc
                
                if print_summary:
                    print('class:', class_num + 1, '/', num_class,
                          ', anon percentage:', p_anon, 
                          ', auroc:', auroc)
        
        self.auroc = auroc_scores
        self.auprc = auprc_scores
                
        self.mean_auroc = np.mean(auroc_scores, axis=0)
        self.mean_auprc = np.mean(auprc_scores, axis=0)

    def display(self):
        if self.auroc is None:
            print('run eval first')
            return
            
        l = [(i,j) for i, j in zip(self.percentiles, self.mean_auroc)]
        print('(anomaly percentage, auroc)')
        print(l)

            

class OneclassEvaluationStatistics():
    def __init__(self, 
                 name = 'unnamed'):
        self.name = name
        self.auroc = None
        self.auprc = None
        self.mean_auroc = None
        self.mean_auprc = None
    
    
    def eval(self, x_train, y_train, x_test, y_test, clf, print_summary=True):
        num_class = np.max(y_train) + 1
        
        auroc_scores = np.zeros([num_class])
        auprc_scores = np.zeros([num_class])

        for class_num in range(num_class):
            
            clf.fit(x_train[y_train==class_num])
            score = clf.score_samples(x_test)
            auroc = roc_auc_score(y_test==class_num, score)
            auprc = average_precision_score(y_test==class_num, score)

            
            auroc_scores[class_num] = auroc
            auprc_scores[class_num] = auprc
            
            if print_summary:
                print('class:', class_num + 1, '/', num_class,
                        ', auroc:', auroc)
    
        self.auroc = auroc_scores
        self.auprc = auprc_scores
                
        self.mean_auroc = np.mean(auroc_scores)
        self.mean_auprc = np.mean(auprc_scores)
            

