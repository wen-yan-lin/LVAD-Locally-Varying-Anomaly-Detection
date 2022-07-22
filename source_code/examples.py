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

def detect_anomalies_in_mnist(interest_class = 9):
    
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

