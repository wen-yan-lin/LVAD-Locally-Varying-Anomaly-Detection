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
from tf_features import folders2generator, generator2feature, download_traditional_network
from normalization import NormalizedPCALearner, NormalizedAnomalyDetector    
from shell_anomaly_detectors import LVAD
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from dataLoader import importData
from evaluation import build_eval_set
from display import display_minst, display_im
import matplotlib.pyplot as plt


def score_reordered_index(given_index, scores):
    chosen_scores = scores[given_index]
    scores_rerank = np.argsort(chosen_scores)
    new_index = given_index[scores_rerank]
    return new_index
     


def simple_example(data_set = 5, interest_class = 0, anomaly_percentage = 0.1):

    x_train, y_train, x_test, y_test, _ = importData(data_set)

    x_data = np.concatenate([x_train, x_test], axis=0)
    y_data = np.concatenate([y_train, y_test])

    data, gt = build_eval_set(x_data, y_data, interest_class, anomaly_percentage)
    lvad = NormalizedAnomalyDetector(clf = LVAD(max_num_clus = 300))
    lvad.fit(data)
    score = lvad.score_samples(data)
    print('AUROC of LVAD:', roc_auc_score(gt, score))

    ocsvm = NormalizedAnomalyDetector(clf = OneClassSVM())
    ocsvm.fit(data)
    score = ocsvm.score_samples(data)
    print('AUROC of OCSVM:', roc_auc_score(gt, score))



def pca_problem(data_set = 5, interest_class = 0, anomaly_percentage = 0.1,
                num_pca_dim = 3, 
                kde_bandwidth = 0.2
                ):

    x_train, y_train, x_test, y_test, _ = importData(data_set)

    x_data = np.concatenate([x_train, x_test], axis=0)
    y_data = np.concatenate([y_train, y_test])

    data, gt = build_eval_set(x_data, y_data, interest_class, anomaly_percentage)


    pca_clf = NormalizedPCALearner(num_dim = num_pca_dim, 
                                    kde_bandwidth = kde_bandwidth)
    pca_clf.fit(data)
    score = pca_clf.score_samples(data)
    print('AUROC of PCA:', roc_auc_score(gt, score))


    lvad = NormalizedAnomalyDetector(clf = LVAD(max_num_clus = 300))
    lvad.fit(data)
    score = lvad.score_samples(data)
    print('AUROC of LVAD:', roc_auc_score(gt, score))

def detect_anomalies_in_mnist_display1(interest_class = 9):
    
    x_train, y_train, x_test, y_test, dataset_name = importData(5)
    print('Interest class:', interest_class)


    x_data = x_train[y_train==interest_class]


    #print('Detecting anomalies with OCSVM')
    ocsvm = NormalizedAnomalyDetector(clf = OneClassSVM())
    ocsvm.fit(x_data)
    scores = ocsvm.score_samples(x_data)
    ranks = np.argsort(scores)

    plt.figure(0)
    display_minst(x_data[ranks[:20]], 
                num_rows=4)
    plt.title('The twenty most anomalous members as determined by OCSVM:')



    #print('Detecting anomalies with LVAD')
    lvad = NormalizedAnomalyDetector(clf = LVAD(max_num_clus=300))
    lvad.fit(x_data)
    scores = lvad.score_samples(x_data)
    ranks = np.argsort(scores)

    plt.figure(1)
    display_minst(x_data[ranks[:20]], 
                num_rows=4)
    plt.title('The twenty most anomalous members as determined by LVAD:')


def detect_anomalies_in_mnist_display2(interest_class = 1):
    num_rows = 5
    max_im_to_display = 100

    x_train, y_train, x_test, y_test, dataset_name = importData(5)
    print('Interest class:', interest_class)


    x_data = x_train[y_train==interest_class]

    ocsvm = NormalizedAnomalyDetector(clf = OneClassSVM())
    ocsvm.fit(x_data)
    ocsvm_scores = ocsvm.score_samples(x_data)
    ocsvm_ranks = np.argsort(ocsvm_scores)


    nLVAD = NormalizedAnomalyDetector(LVAD(max_num_clus=300))
    nLVAD.fit(x_data)
    nLVAD_scores = nLVAD.score_samples(x_data)
    nLVAD_ranks = np.argsort(nLVAD_scores)

    steps = x_data.shape[0] // max_im_to_display
    dis_data = x_data[ocsvm_ranks[::steps]]
    plt.figure(0)
    display_minst(dis_data[:max_im_to_display], 
                num_rows=num_rows,
                print_file = 'ocsvm_' + str(interest_class)+'.png')
    plt.title('Anomaly ranking as determined by OCSVM:')

    # ensure that both LVAD and OCSVM display the same set of images
    match_nlVAD_index = score_reordered_index(ocsvm_ranks[::steps], nLVAD_scores)
    dis_data = x_data[match_nlVAD_index]
    plt.figure(1)
    display_minst(dis_data[:max_im_to_display], 
                num_rows=num_rows,
                print_file = 'lvad_' + str(interest_class)+'.png')
    plt.title('Anomaly ranking as determined by LVAD:')

    print('High resolution images are saved in folder')



def anomalies_in_folder(root_folder_name, 
                        sub_folder_number = 0, 
                        max_im_to_display = 100,
                        num_rows_to_display = 5,
                        batch_size = 16
                        ):

    gen = folders2generator(root_folder_name, batch_size = batch_size, target_class=[sub_folder_number])
    resNet = download_traditional_network()
    
    feats, gts, im_paths = generator2feature(resNet, gen)
    feats = np.concatenate(feats, axis=0)
    gts = np.concatenate(gts)
    im_paths = sum(im_paths, [])
    full_paths = [root_folder_name + '/' + im_path for im_path in im_paths]

    nlvad = NormalizedAnomalyDetector(clf = LVAD())
    nlvad.fit(feats)
    scores = nlvad.score_samples(feats)
    
    ranks = np.argsort(scores)
    steps = ranks.size // max_im_to_display
    chosen_index = ranks[::steps]
    chosen_paths = [full_paths[ind] for ind in chosen_index]
    plt.figure(0)
    display_im(chosen_paths, 
            num_rows = num_rows_to_display, 
            print_file = 'lvad.jpg')
    plt.title('Overall Anomaly ranking by LVAD')

    plt.figure(1)
    chosen_index = ranks[:50]
    chosen_paths = [full_paths[ind] for ind in chosen_index]
    display_im(chosen_paths, 
            num_rows = 5, 
            print_file = 'lvad-worst-50.jpg')
    plt.title('Top 50 Anomalies')

