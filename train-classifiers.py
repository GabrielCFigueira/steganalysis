import sys
import os
import argparse
import pathlib
import glob
import magic
import subprocess
import re
import math
import json
import csv
import joblib

import fleep
import pandas
import numpy as np
import random
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import multiprocessing
import threading
# --------------------------------------------

# CLASSES

# --------------------------------------------


class File:
    """
    Attributes:
        file_name: A string containing the file name
        file_type: A string containing the file type (image, video, other)
        file_extension: A string containing the file extension
        file_size: A float containing the size of the file in bytes
        features: A dict containing each type of feature from file for ML -> {feature_set_name: [features]}
        classification: A dict containing of structure { classifier : prediction, etc }
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.file_type = ''
        self.file_extension = ''
        self.file_size = ''
        self.features = None
        self.classification = {}

    def set_file_type(self, file_type):
        self.file_type = file_type

    def set_file_extension(self, file_extension):
        self.file_extension = file_extension

    def set_file_size(self, file_size):
        self.file_size = file_size

    def add_features(self, feature_source, feature_list):
        self.features[feature_source] = feature_list

    def update_file(self, file_type, file_extension, file_size):
        self.file_type = file_type
        self.file_extension = file_extension
        self.file_size = file_size

    def set_classification(self, classifier, prediction):
        self.classification[classifier] = prediction

def fileName(string):
    split = string.split("_")
    return split[0], int(split[1][1:])

def aggregate(data):

    file_name, _ = fileName(data['file_name'][0])
    current_file_name = file_name

    n_videos = 1
    n_frames = 300 #TODO improve this
    oldI = 300
    for idx in range(len(data.index)):
        current_file_name, i = fileName(data['file_name'][idx])
    
        if current_file_name != file_name:
            n_videos += 1
            file_name = current_file_name
            if oldI < n_frames:
                n_frames = oldI

        oldI = i
    
    print("n_videos: " + str(n_videos))
    print("n_frames: " + str(n_frames))

    n_columns = len(data.iloc[0]) - 2
    columns = []
    for i in range(n_frames):
        for j in range(n_columns):
            columns += [str(i * n_frames + j)]
    
    columns += ['class']

    file_name, _ = fileName(data['file_name'][0])
    current_file_name = file_name

    newData = np.ndarray(shape=(n_videos,n_columns * n_frames + 1), dtype=float)

    video_id = 0
    for idx in range(len(data.index)):
        current_file_name, i = fileName(data['file_name'][idx])

        if current_file_name != file_name:
            file_name = current_file_name
            newData[video_id][n_columns * n_frames] = data['class'][idx - 1]
            video_id += 1
        elif i > n_frames:
            continue

        row = data.iloc[idx].drop(['class', 'file_name'])
        for d in range(len(row)):
            newData[video_id][(i - 1) * n_columns + d] = row[d]

    newData[video_id][n_columns * n_frames] = data['class'][len(data.index) - 1]
    newData = pandas.DataFrame(data=newData, columns=columns)
    return newData, columns


def computeNewFeatures(newData, video_id, features):
    n_features = 17 # number of statistical features

    for i in range(len(features)):
        newData[video_id][i * n_features] = np.mean(features[i])                                                                                                           
        newData[video_id][i * n_features + 1] = np.median(features[i])                                                                                                     
        newData[video_id][i * n_features + 2] = np.std(features[i])                                                                                                        
        newData[video_id][i * n_features + 3] = np.var(features[i])                                                                                                        
        newData[video_id][i * n_features + 4] = kurtosis(features[i])                                                                                                      
        newData[video_id][i * n_features + 5] = skew(features[i])                                                                                                          
        newData[video_id][i * n_features + 6] = np.amax(features[i])                                                                                                       
        newData[video_id][i * n_features + 7] = np.amin(features[i])
        newData[video_id][i * n_features + 8] = np.percentile(features[i], 10)
        newData[video_id][i * n_features + 9] = np.percentile(features[i], 20)
        newData[video_id][i * n_features + 10] = np.percentile(features[i], 30)
        newData[video_id][i * n_features + 11] = np.percentile(features[i], 40)
        newData[video_id][i * n_features + 12] = np.percentile(features[i], 50)
        newData[video_id][i * n_features + 13] = np.percentile(features[i], 60)
        newData[video_id][i * n_features + 14] = np.percentile(features[i], 70)
        newData[video_id][i * n_features + 15] = np.percentile(features[i], 80)
        newData[video_id][i * n_features + 16] = np.percentile(features[i], 90)

def statistics(data):
    
    file_name, _ = fileName(data['file_name'][0])
    current_file_name = file_name

    n_videos = 1
    for idx in range(len(data.index)):
        current_file_name, i = fileName(data['file_name'][idx])
    
        if current_file_name != file_name:
            n_videos += 1
            file_name = current_file_name
    
    
    print("n_videos: " + str(n_videos))

    n_columns = len(data.iloc[0]) - 2
    n_features = 17 # number of statistical features
    columns = []
    for j in range(n_columns):
        columns += ["mean@" + str(j)]
        columns += ["median@" + str(j)]
        columns += ["std@" + str(j)]                                                 
        columns += ["var@" + str(j)]
        columns += ["kurtosis@" + str(j)]
        columns += ["skew@" + str(j)]                        
        columns += ["max@" + str(j)]
        columns += ["min@" + str(j)]
        columns += ["p10@" + str(j)]                    
        columns += ["p20@" + str(j)]     
        columns += ["p30@" + str(j)]                                                                      
        columns += ["p40@" + str(j)]                     
        columns += ["p50@" + str(j)]                                    
        columns += ["p60@" + str(j)]               
        columns += ["p70@" + str(j)]
        columns += ["p80@" + str(j)]          
        columns += ["p90@" + str(j)]
    columns += ['class']


    file_name, _ = fileName(data['file_name'][0])
    current_file_name = file_name

    newData = np.ndarray(shape=(n_videos,n_columns * n_features + 1), dtype=float)
    
    features = [[]] * n_columns
    video_id = 0
    for idx in range(len(data.index)):
        current_file_name, _ = fileName(data['file_name'][idx])

        if current_file_name != file_name:
            file_name = current_file_name

            computeNewFeatures(newData, video_id, features)
            newData[video_id][n_columns * n_features] = data['class'][idx - 1]
            video_id += 1
            features = [[]] * n_columns

        row = data.iloc[idx].drop(['class', 'file_name'])
        for d in range(len(row)):
            features[d] = features[d] + [row[d]]
    computeNewFeatures(newData, video_id, features)
    newData[video_id][n_columns * n_features] = data['class'][len(data.index) - 1]
    newData = pandas.DataFrame(data=newData, columns=columns)
    return newData, columns
# --------------------------------------------

# FUNCTIONS

# --------------------------------------------

def create_xgb_classifier(file_type):
    csv_file = 'vid-features.csv'
    joblib_file = 'vid-xgb.joblib'

    print('=== Handling SVM for {} files ... ==='.format(file_type))

    print('[*] Reading {} ... '.format(csv_file))
    training_data = pandas.read_csv(csv_file)
   
    columns = training_data.columns
    training_data, columns = statistics(training_data)

    training_data = training_data.sample(frac=1)
    print('[*] Getting x and y ... ')
    x = training_data.drop(['class'], axis=1)
    y = training_data['class']

    model = XGBClassifier()
    cv = StratifiedKFold(n_splits=10)
    tprs = []
    aucs = []
    importances = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train, test in cv.split(x, y):

        model = model.fit(np.asarray(x)[train], np.asarray(y)[train])

        probas_ = model.predict_proba(np.asarray(x)[test])

        fpr, tpr, thresholds = roc_curve(np.asarray(y)[test], probas_[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        if(roc_auc < 0.5):
            roc_auc = 1 - roc_auc
            fpr = [1 - e for e in fpr]
            fpr.sort()
            tpr = [1 - e for e in tpr]
            tpr.sort()
        print("roc_auc: " + str(roc_auc))

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)

        f_imp = model.feature_importances_
        importances.append(f_imp)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    if(mean_auc < 0.5):
        mean_auc = 1 - mean_auc
        fpr = [1 - e for e in fpr]
        fpr.sort()
        tpr = [1 - e for e in tpr]
        tpr.sort()

    print("[*] Model AUC: " + "{0:.3f}".format(mean_auc))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    std_auc = np.std(aucs)
    
    np.save("ROC_10CV_XGBoost_Sensitivity", np.array(mean_tpr))
    np.save("ROC_10CV_XGBoost_Specificity", np.array(mean_fpr))

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'$\pm$ ROC Std. Dev.')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='0.0', label='Random Guess')
    ax1.yaxis.grid(color='black', linestyle='dotted', lw=0.2)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.legend(loc='lower right', frameon=False, handlelength=1.0, fontsize=14)

    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)

    ax1.set(xlim=(0, 1), ylim=(0.0, 1))
    plt.tight_layout()
    
    fig.savefig('Steganalysis.pdf')
    plt.close(fig)

    #Compute mean importance of feature accross CV folds
    bin_number = list(range(len(x.iloc[0])))
    mean_importances = []                                          

    for n in range(0,len(importances[0])):                                           
    
        mean_imp = (importances[0][n] + importances[1][n] + importances[2][n] + importances[3][n] + importances[4][n] + importances[5][n] + importances[6][n] + importances[7][n] + importances[8][n] + importances[9][n])/10.0
        mean_importances.append(mean_imp)

    #print mean_importances
    f_imp = zip(bin_number,mean_importances, columns)
    f_new = sorted(f_imp, key = lambda t: t[1], reverse=True)
    np.save('weights', np.array(f_new))


# --------------------------------------------



# FUNCTION: WRITE TO CSV
def write_vid_csv(stego_files_features, clean_files_features):
    # set file name
    output_file = 'vid-features.csv'
    
    resultingDF = pandas.DataFrame()
    for file in stego_files_features:
        file.features['class'] = [1] * len(file.features.index)
        resultingDF = resultingDF.append([file.features])
    # update class of clean files
    for file in clean_files_features:
        file.features['class'] = [0] * len(file.features.index)
        resultingDF = resultingDF.append([file.features])
        
    resultingDF.to_csv(output_file)
        
    # update user again
    print('[*] Extracted video features can be found in {}.'.format(output_file))


# --------------------------------------------


# FUNCTION: GET IDFB FEATURES
def get_idfb_features(file, thread_id):
    # set up files for bash cmds
    input_file = file.file_name
    output_file = 'temp-features' + str(thread_id) + '.csv'
    extractor = 'IDFB/extractor.exe'
    bash_cmd = 'wine {} -s -t 1 -i {} -o {}'.format(extractor, input_file, output_file)

    if os.path.exists(output_file):
        os.remove(output_file)
    print('... Calling subprocess ')
    video_extraction_process = subprocess.Popen([bash_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    output, error = video_extraction_process.communicate()
    decoded_output = output.decode('utf-8')

    # set up column names for pandas
    col_names = []
    for i in range(768):
        col_i = 'IDFB_{}'.format(i + 1)
        col_names.append(col_i)

    # get data from csv
    temp_csv = pandas.read_csv(output_file, sep=' ', names=col_names, index_col=False)
    expected_lines = len(temp_csv.index)
    # set up row names for pandas
    row_names = []
    for i in range(expected_lines):
        row_i = '{}_f{}'.format(input_file, i + 1)
        row_names += [row_i]

    temp_csv['file_name'] = row_names

    # remove temp-features.csv
    if os.path.exists(output_file):
        os.remove(output_file)

    file.features = temp_csv

    return file


# FUNCTION: GET SUPERB FEATURES
def get_superb_features(file, thread_id):
    # set up files for bash cmds
    input_file = file.file_name
    output_file = 'temp-features' + str(thread_id) + '.csv'
    extractor = 'SUPERB/extractor.exe'
    bash_cmd = 'wine {} -s -i {} -o {}'.format(extractor, input_file, output_file)

    if os.path.exists(output_file):
        os.remove(output_file)
    print('... Calling subprocess ')
    video_extraction_process = subprocess.Popen([bash_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    output, error = video_extraction_process.communicate()
    decoded_output = output.decode('utf-8')

    # set up column names for pandas
    col_names = []
    for i in range(609):
        col_i = 'SUPERB_{}'.format(i + 1)
        col_names.append(col_i)

    # get data from csv
    temp_csv = pandas.read_csv(output_file, sep=' ', names=col_names, index_col=False)
    expected_lines = len(temp_csv.index)
    # set up row names for pandas
    row_names = []
    for i in range(expected_lines):
        row_i = '{}_f{}'.format(input_file, i + 1)
        row_names += [row_i]

    temp_csv['file_name'] = row_names

    # remove temp-features.csv
    if os.path.exists(output_file):
        os.remove(output_file)

    file.features = temp_csv

    return file


def thread_steganalysis(file_list, thread_id, total_threads):

    for i in range(thread_id, len(file_list), total_threads):
        print('[*] {} of {} files'.format(i, len(file_list)))
        file = file_list[i]
        if file.file_type == 'video':
            file = get_idfb_features(file, thread_id)


# FUNCTION: PERFORM STEGANALYSIS
def perform_steganalysis(file_list, group_type):
    # update user on progress
    print('\n=== Performing feature extraction on {} files (this will take a while) ... ==='.format(group_type))
    # get features for each file
    total_threads = multiprocessing.cpu_count()
    threads = []
    for i in range(total_threads):
        thread = threading.Thread(target=thread_steganalysis, args=(file_list, i, total_threads))
        thread.start()
        threads += [thread]
    for thread in threads:
        thread.join()

    # update user again
    print('=== Steganalysis complete! ===')
    # return files
    return file_list


# --------------------------------------------


# FUNCTION: GET FILE TYPE OF INPUT FILE
def get_file_type(file_name):
    with open(file_name, 'rb') as file:
        file_info = fleep.get(file.read(128))
    if file_info.type_matches('video'):
        file_type = 'video'
        file_extension = file_info.extension[0]
    else:
        h264_flag = 'H.264'
        magic_info = magic.from_file(file_name)
        if h264_flag in magic_info:
            file_type = 'video'
            file_extension = 'h264'
        else:
            file_type = 'other'
            file_extension = pathlib.Path(file_name).suffix  # get file extension from pathlib instead
    return file_type, file_extension


# FUNCTION: FIND INPUT FILE IN FILESYSTEM
def find_file(file_name):
    if os.path.isfile(file_name):
        return True
    else:
        return False


# FUNCTION: GET LIST OF FILES
def get_file_lists(dir_location):
    file_names = glob.glob("{}/*".format(dir_location))
    file_list = []
    for file_name in file_names:
        if find_file(file_name):  # try to find file, if file can be found:
            new_file = File(file_name)  # create File object
            file_type, file_extension = get_file_type(file_name)  # get file type and file extension
            if file_type != 'other':
                file_size = os.path.getsize(file_name)  # get file size
                new_file.update_file(file_type, file_extension, file_size)  # update new_file with new info
                file_list.append(new_file)  # add file object to file list
    return file_list


# --------------------------------------------


# FUNCTION: FEATURE EXTRACTION
def extract_features(dir_location, file_type):
    # get file lists of File objects
    stego_files = get_file_lists("{}/stego".format(dir_location))
    clean_files = get_file_lists("{}/clean".format(dir_location))
    print('[*] Number of stego {} files: {}'.format(file_type, len(stego_files)))
    print('[*] Number of clean {} files: {}'.format(file_type, len(clean_files)))
    # get features for stego files
    stego_files_features = perform_steganalysis(stego_files, 'stego')
    # get features for clean files
    clean_files_features = perform_steganalysis(clean_files, 'clean')
    # return files
    return stego_files_features, clean_files_features


# FUNCTION: RUN PROGRAM
def run(dir_location):
    # get dir paths
    vid_dir = "{}/videos".format(dir_location)
    if not os.path.isfile("vid-features.csv"):
        # extract features
        print('\n===== EXTRACTING VIDEO FEATURES =====\n')
        vid_stego_features, vid_clean_features = extract_features(vid_dir, 'video')
        # write to csvs
        print('\n===== WRITING FEATURES TO DISK =====\n')
        write_vid_csv(vid_stego_features, vid_clean_features)
    # create & train svm
    #print('\n===== CREATING & TRAINING SVMs =====\n')
    #create_svm_classifier('video')
    #print('\n===== CREATING & TRAINING LOGISTIC REGRESSION CLASSIFIERS =====\n')
    #create_lr_classifier('video')
    print('\n===== CREATING & TRAINING XGBOOST CLASSIFIERS =====\n')
    create_xgb_classifier('video')


# MAIN FUNCTION: GLOBAL VARIABLES
if __name__ == '__main__':

    warnings.filterwarnings(action='ignore')

    # argument parsing
    parser = argparse.ArgumentParser(description='A script to extract image & video features, '
                                                 '& train machine learning classifiers [SVM & Logistic Regression]. ')
    parser.add_argument('dir_location', action="store", help='Directory location of training data in quotation marks')
    args = parser.parse_args()
    # handle arguments
    print('Searching for directory ...')
    if args.dir_location:
        if os.path.exists(args.dir_location):
            print('Location found!')
            run(args.dir_location)
        else:
            print('Location not found!')
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
