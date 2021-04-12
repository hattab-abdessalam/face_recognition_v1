import cv2
import numpy as np
import os
import glob

import scipy
def chi_square(h1, h2):
    h1, h2 = __prepare_histogram(h1, h2)
    old_err_state = np.seterr(
        invalid='ignore')  # divide through zero only occurs when the bin is zero in both histograms, in which case the division is 0/0 and leads to (and should lead to) 0
    result = np.square(h1 - h2) / (h1 + h2)
    np.seterr(**old_err_state)
    result[np.isnan(result)] = 0  # faster than scipy.nan_to_num, which checks for +inf and -inf also
    return np.sum(result)


def __prepare_histogram(h1, h2):
    """Convert the histograms to scipy.ndarrays if required."""
    h1 = h1 if scipy.ndarray == type(h1) else scipy.asarray(h1)
    h2 = h2 if scipy.ndarray == type(h2) else scipy.asarray(h2)
    if h1.shape != h2.shape or h1.size != h2.size:
        raise ValueError('h1 and h2 must be of same shape and size')
    return h1, h2

def conv_list_arr(histogrammes,nd,sk):


    histo=[]
    for i in range(nd):
        for j in range (sk):
            t = np.array(histogrammes[i][j])
            t=t.astype('float32')

            histo.append(t)

    return histo
def read_images(path):
    nomder = len(os.listdir(path ))
    print(nomder)
    total=0
    for i in range(1,nomder+1):
        n=str(i)
        nombimg = len(glob.glob1(path + '/' + n + '/', '*'))
        total=total+nombimg
    for ImageName in os.listdir(path + '/' + '1'):
        Image_path = os.path.join(path + '/' + '1', ImageName)
        img = cv2.imread(Image_path, 0)
        break
    #img = cv2.resize(img, (128, 128))

    sifthi=[]
    for name in range(1, nomder+1):
        name = str(name)
        print('import directory ' + name)
        for ImageName in os.listdir(path+'/' + name):
            Image_path = os.path.join(path+'/'+ name, ImageName)
            img = cv2.imread(Image_path, 0)
            #img = cv2.resize(img, (128, 128))
            sift = cv2.xfeatures2d.SIFT_create()

            kp1, des1 = sift.detectAndCompute(img, None)
            sifthi.append(des1)
    return sifthi
if __name__ == '__main__':

    data=read_images("D://datasets//face//crossvalidation//data//orl//orl1//train")
    np.save('sift_orl1_train.npy', data)
    data = np.load('sift_orl1_train.npy', allow_pickle=True)
    data2 = read_images('D://datasets//face//crossvalidation//data//orl//orl1//test')
    np.save('sift_orl1_test.npy', data2)
    data2 = np.load('sift_orl1_test.npy', allow_pickle=True)



    '''FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    a = np.load('sift_orl2_train_histo_altp_train.npy', allow_pickle=True)
    data = conv_list_arr(a, 40, 8)
    b = np.load('sift_orl2_test_histo_altp_test.npy', allow_pickle=True)
    data2 = conv_list_arr(b, 40, 2)'''


    for j in range(80):

        total = []
        for i in range(320):

            '''indx = 0
            for k in range(0,data2[j].shape[0]):
                # ind = np.linalg.norm(histogrammes[i] - histogramme_test[ii][j][k], axis=1).min(
                mat_chi_square = np.zeros(data[i].shape[0])
                for kk in range(data[i].shape[0]):
                    a = chi_square(data[i][kk], data2[j][k])
                    mat_chi_square[kk] = a
                ind = mat_chi_square.min()
                #
                indx = indx + ind
            total.append(indx)
        t = np.array(total)
        if(t.argmin()//8==j//2):
            print('b1')
        else:
            print('erreur')'''
            if data2[j] is None:
                print('false')
                break
            else:

                #matches = flann.knnMatch(data[i], data2[j], k=2)
                bf=cv2.BFMatcher()


                matches = bf.knnMatch(data[i], data2[j], k=2)

                k = 0
                for m, n in matches:

                    if m.distance < 0.8* n.distance:
                        k = k + 1

                total.append(k)
        if data2[j] is not None:
            t = np.array(total)
            if (t.argmax() // 8 == j // 2):
                print('b1')
            else:
                print('erreur')