# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 05:59:16 2020

@author: BS
"""

import numpy as np
from tqdm import tqdm
import glob
import os
import cv2
from joblib import Parallel, delayed
import multiprocessing
import fonctions
import fonctions_yal
from datetime import datetime
from sklearn import preprocessing
import scipy
import ALTP


def chi_square(h1, h2):
    h1, h2 = __prepare_histogram(h1, h2)
    old_err_state = scipy.seterr(
        invalid='ignore')  # divide through zero only occurs when the bin is zero in both histograms, in which case the division is 0/0 and leads to (and should lead to) 0
    result = scipy.square(h1 - h2) / (h1 + h2)
    scipy.seterr(**old_err_state)
    result[scipy.isnan(result)] = 0  # faster than scipy.nan_to_num, which checks for +inf and -inf also
    return scipy.sum(result)


def __prepare_histogram(h1, h2):
    """Convert the histograms to scipy.ndarrays if required."""
    h1 = h1 if scipy.ndarray == type(h1) else scipy.asarray(h1)
    h2 = h2 if scipy.ndarray == type(h2) else scipy.asarray(h2)
    if h1.shape != h2.shape or h1.size != h2.size:
        raise ValueError('h1 and h2 must be of same shape and size')
    return h1, h2


def histogramme(type, image):
    h = np.zeros((256))
    if (type == 1):
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                valeur = image[j, i]
                h[int(valeur)] += 1
    else:

        for line in range(1, np.shape(image)[0] - 1):
            for column in range(1, np.shape(image)[1] - 1):
                bloc = image[line - 1:line - 1 + 3, column - 1:column - 1 + 3]
                for j in range(bloc.shape[0]):
                    for i in range(bloc.shape[1]):
                        valeur = bloc[j, i]
                        h[int(valeur)] += 1
    return h


def nomb_img(path):
    nomder = len(os.listdir(path))
    print(nomder)
    total = 0
    for i in range(1, nomder + 1):
        n = str(i)
        nombimg = len(glob.glob1(path + '/' + n + '/', '*'))
        total = total + nombimg
    return total


def conv_list_arr(histogrammes, nbr_img):
    histo = np.zeros((nbr_img, histogrammes[0].shape[1]), dtype='int')
    total = len(histogrammes)
    k = 0
    for i in range(total):
        a = histogrammes[i]
        for j in range(a.shape[0]):
            histo[k + j, :] = np.array(a[j]).flatten()
        k = a.shape[0] + k

    return histo


def calculate(type, n, path_faces):  # calculer les histogrammes ltp du dossier n qui trouve dans le lien path_faces

    derName = str(n)  # convertir le nom du dossier en string
    name = str(n)
    i = 0
    t = 5  # threshold
    nombimg = len(
        glob.glob1(path_faces + '/' + derName + '/', '*'))  # calculer le nombre des visages dans le dossier derName
    histo = np.zeros((nombimg, 512), dtype='int')  # matrice pour sauvegarder les histogrammes lbp du dossier derName

    for ImageName in os.listdir(path_faces + '/' + name):
        Image_path = os.path.join(path_faces + '/' + name, ImageName)
        print(Image_path)
        img = cv2.imread(Image_path, 0)
        # img = cv2.resize(img, (371,243 ))

        matrice_ltp1, matrice_ltp2 = calculate_matrice_LTP(img, t)  # calculer matrice LTP de (img )
        h1 = histogramme(type, np.array(matrice_ltp1))  # calculer histogramme ltp upper
        h2 = histogramme(type, np.array(matrice_ltp2))  ##calculer histogramme ltp lower
        hi = np.concatenate((h1, h2))
        histo[i, :] = hi  # Sauvegarder histogramme dans la matrice des histogrammes
        print(name, '-------->', i)  # afficher le nombre des images déjà traité
        i += 1
    return histo


def calculate_matrice_LTP(image, t):
    img_LTP1 = np.zeros((np.shape(image)[0] - 2, np.shape(image)[1] - 2))  # matrice lTp upper -2 puisque LTP(3*3)
    img_LTP2 = np.zeros((np.shape(image)[0] - 2, np.shape(image)[1] - 2))  # matrice ltp lower -2 puisque LTP(3*3)
    for line in range(np.shape(image)[0] - 2):
        for column in range(np.shape(image)[1] - 2):
            img_LTP1[line, column], img_LTP2[line, column] = calculateLTP(image, column + 1, line + 1,
                                                                          t)  # calculer les codes ltp upper,ltp lower  pour le pixel (line,column)

    return img_LTP1, img_LTP2


def calculateLTP(image, column, line, t):
    neighbours = get_neighbours(image, column, line)
    center = np.array(image)[line, column]
    values1, values2 = calculate_code(center, neighbours, t)  # calculer les codes binaire ltp upper ,ltp lower
    # convertir les codes binaire en décimale
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    LTP1 = 0
    LTP2 = 0
    for i in range(0, len(values1)):
        LTP1 += values1[i] * weights[i]
        LTP2 += values2[i] * weights[i]
    return LTP1, LTP2


def get_neighbours(image, column, line):  # obtenir les voisins du pixel (column,line)
    bloc = image[line - 1:line - 1 + 3, column - 1:column - 1 + 3]
    a = np.array(bloc).flatten()
    neighbours = [a[0], a[1], a[2], a[5], a[8], a[7], a[6], a[3]]
    return neighbours


def calculate_code(center, neighbours, t):  # calculer les codes binaire

    low = center - t
    high = center + t

    result = []
    for neighbour in neighbours:
        if neighbour >= high:
            result.append(1)
        else:
            if neighbour <= low:
                result.append(-1)
            else:
                result.append(0)

    lTPcode1 = np.copy(result)
    lTPcode2 = np.copy(result)

    for i in range(0, 8):  # ltp upper (modifier -1 par un 0)
        if lTPcode1[i] == -1:
            lTPcode1[i] = 0

    for i in range(0, 8):  # ltp lower (modifier 1 par un 0 et le -1 par un 1)
        if lTPcode2[i] == 1:
            lTPcode2[i] = 0
        if lTPcode2[i] == -1:
            lTPcode2[i] = 1

    return lTPcode1, lTPcode2


if __name__ == '__main__':
    print('LTP')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)  # afficher l'heure de démarrage
    num_cores = multiprocessing.cpu_count()  # nombre des coeurs cpu
    print(num_cores)

    train_path = "sift_orl_train"  # lien du dossier (train)
    nbder_train = len(os.listdir(train_path))  # nombre des dossiers (personnes) dans le dossier train
    histo=[]
    for i in range(1,nbder_train+1):
        n_train=len(os.listdir(train_path+'\\'+str(i)))#nombre des dossiers (personnes)
        histogrammes= Parallel(n_jobs=num_cores)(delayed(ALTP.calculate)(1,kl, train_path+'\\'+str(i)) for kl in tqdm(range(1, n_train+1)))#calculer histogrammes ltp (train)
        histogrammes_train=conv_list_arr(histogrammes,nomb_img( train_path+'\\'+str(i)))#convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante

        histo.append(histogrammes)
    np.save('sift_orl_train_histo_yolo_ltp_train.npy', histo )#souvgarder les histogrammes dans le disque dur'''
    histogrammes = np.load('sift_orl_train_histo_yolo_ltp_train.npy', allow_pickle=True)

    test_path = "sift_orl_test"  # lien du dossier (train)
    nbder_test = len(os.listdir(test_path))  # nombre des dossiers (personnes) dans le dossier train

    histo_test=[]
    for i in range(1,nbder_test+1):
        n_test=len(os.listdir(test_path+'\\'+str(i)))#nombre des dossiers (personnes)
        histogrammes= Parallel(n_jobs=num_cores)(delayed(ALTP.calculate)(1,kl, test_path+'\\'+str(i)) for kl in tqdm(range(1, n_test+1)))#calculer histogrammes ltp (train)
        #histogrammes_test=conv_list_arr(histogrammes,nomb_img( test_path+'\\'+str(i)))#convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante

        histo_test.append(histogrammes)
    np.save('sift_orl_test_histo_yolo_ltp_test.npy', histo_test )#souvgarder les histogrammes dans le disque dur'''

    histogramme_test = np.load('sift_orl_test_histo_yolo_ltp_test.npy', allow_pickle=True)

    '''print(histogramme_test[0][0].shape)
    acc=0
    for ii in range (0,40):   
        for j in range (0,3):
            total=[]
            for i in range(0,40):
                indx=0
                for k in range(0,histogramme_test[ ii][j].shape[0]):
                    ind = np.linalg.norm(histogrammes[i] - histogramme_test[ii][j][k], axis=1).min()
                    indx=indx+ind
                total.append(indx)
            totalar=np.array(total)


            if (totalar.argsort()[0]==(ii*3+j)//3):
                acc=acc+1
                print(acc)

                    mat_chi_square = np.zeros(histogrammes_lairn.shape[0])
                for ii in range(histogrammes_lairn.shape[0]):
                    a = chi_square(histogrammes_lairn[ii], histogrammes_test[ind])
                    mat_chi_square[ii] = a
                index = mat_chi_square.argsort()'''
    print(histogrammes[1][1].shape)
    print(histogramme_test[1][1].shape)
    acc = 0
    total = []
    for ii in range(0, 40):
        for j in range(0, 3):

            total=[]
            for i in range(0, 40):
                for kk in range(0,7):
                    indx=0
                    for k in range(0, histogramme_test[ii][j].shape[0]):
                        mat_chi_square = np.zeros(histogrammes[i][kk].shape[0])
                        for ss in range(histogrammes[i][kk].shape[0]):
                            a = chi_square(histogrammes[i][kk][ss], histogramme_test[ii][j][k])
                            mat_chi_square[ss] = a
                        ind = mat_chi_square.min()
                        #ind= np.linalg.norm(histogrammes[i][kk] - histogramme_test[ii][j][k], axis=1).min()
                    indx=indx+ind
                    total.append(indx)
            totalar = np.array(total)
            to=totalar.argsort()
            if (totalar.argsort()[0]//7==ii):
                acc=acc+1
                print(acc)
            else:
                print('erreur',ii,j)


