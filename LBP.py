import numpy as np
from tqdm import tqdm
import glob
import os
import cv2
from joblib import Parallel, delayed
import multiprocessing
import fonctions
from datetime import datetime
import fonctions_yal
from sklearn import preprocessing
def calculate(type,n, path_faces): #calculer les histogrammes lbp du dossier n qui trouve dans le lien path_faces
    derName = str(n) # convertir le nom du dossier en string
    name = str(n)
    i = 0
    nombimg = len(glob.glob1(path_faces + '/' + derName + '/', '*'))#calculer le nombre des visages dans le dossier derName
    histo = np.zeros((nombimg, 256), dtype='int') #matrice pour sauvegarder les histogrammes lbp du dossier derName

    for ImageName in os.listdir(path_faces + '/' + name):
        Image_path = os.path.join(path_faces + '/' + name, ImageName)
        img = cv2.imread(Image_path, 0)
        img = cv2.resize(img, (128,128 ))

        matrice_lbp = calculate_matrice_lbp(img) #calculer matrice lbp de (img )
        hi = fonctions.histogramme(type,matrice_lbp) #calculer histogramme lbp
        histo[i, :] = np.array(hi).flatten() #Sauvegarder histogramme dans la matrice des histogrammes
        print(name,'-------->', i)  #afficher le nombre des images déjà traité
        i += 1

    return histo


def calculate_matrice_lbp(image):
    img_lbp = np.zeros((np.shape(image)[0] - 2, np.shape(image)[1] - 2)) #matrice lbp -2 puisque LBP(3*3)
    for line in range(np.shape(image)[0] - 2):
        for column in range(np.shape(image)[1] - 2):
            img_lbp[line, column] = calculateLBP(image, column + 1, line + 1) #calculer le code  lbp pour le pixel (line,column)
    return img_lbp


def calculateLBP(image, column, line):
    neighbours = get_neighbours(image, column, line)
    center = np.array(image)[line, column]
    values = calculate_biniry_code(center, neighbours) #calculer le code binaire lbp
    # convertir le code binaire en décimale
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    lbp = 0
    for i in range(0, len(values)):
        lbp += values[i] * weights[i]
    return lbp


def get_neighbours(image, column, line): #obtenir les voisins du pixel (column,line)
    bloc = image[line - 1:line - 1 + 3, column - 1:column - 1 + 3]
    a = np.array(bloc).flatten()
    neighbours = [a[0], a[1], a[2], a[5], a[8], a[7], a[6], a[3]]
    return neighbours


def calculate_biniry_code(center, neighbours): #calculer le code binaire
    result = []
    for neighbour in neighbours:
        if neighbour >= center:
            result.append(1)
        else:
            result.append(0)
    return result



if __name__ == '__main__':



    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)    #afficher l'heure de démarrage
    train_path = "D://datasets//crossvalidation//data//feret_gray/feret_gray5//train"  # lien du dossier (train)
    test_path = "D://datasets//crossvalidation//data//feret_gray//feret_gray5//test"  # lien du dossier (test)
    xls_path = 'orl_train'
    exel_line=6 #Numéro de la ligne dans le fichier Excel pour écrire les résultats de la reconnaissance

    nbder_train=len(os.listdir(train_path))#nombre des dossiers (personnes) dans le dossier train

    num_cores = multiprocessing.cpu_count()#nombre des coeurs cpu

    histogrammes= Parallel(n_jobs=num_cores)(delayed(calculate)(1,kl, train_path) for kl in tqdm(range(1,nbder_train+1)))#calculer histogrammes lbp (train)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)#l'heure de fin

    np.save('orl_histo_lbp_train.npy', histogrammes )#souvgarder les histogrammes dans le disque dur
    histogrammes_train = np.load('orl_histo_lbp_train.npy',allow_pickle=True)
    print(histogrammes_train.shape)

    histogrammes_train=fonctions.conv_list_arr(histogrammes_train,fonctions.nomb_img( train_path))#convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante


    label_train = fonctions.labels(train_path)#calculer labels train


    nbder_test = len(os.listdir(test_path))#nombre des dossiers (personnes) dans le dossier test
    histogrammes_test= Parallel(n_jobs=num_cores)(delayed(calculate)(1,kl, test_path) for kl in tqdm(range(1, nbder_test+1)))#calculer histogrammes lbp (test)

    np.save('orl_histo_lbp_test.npy', histogrammes_test )#souvgarder les histogrammes dans le disque dur
    histogrammes_test = np.load('orl_histo_lbp_test.npy',allow_pickle=True)
    print(histogrammes_test.shape)

    histogrammes_test=fonctions.conv_list_arr(histogrammes_test,fonctions.nomb_img( test_path)) #convertir  histogrammes_test en 2 dimensions puisque chaque coeur calculer une matrice dépendante

    '''mix=np.concatenate((histogrammes_train,histogrammes_test),axis=0)
    prepare=preprocessing.MinMaxScaler()
    x=prepare.fit_transform(mix.astype(float))
    histogrammes_train=x[0:histogrammes_train.shape[0],:]
    histogrammes_test=x[histogrammes_train.shape[0]:,:]'''

    label_test = fonctions.labels(test_path) #calculer labels test pour vérifer la reconnaissance
    nom_file_excel = 'knn_' + xls_path[:-6] + '.xlsx' #nom du fichier Excel qui est utilisé pour écrire les résultats de la reconnaissance
    Parallel(n_jobs=num_cores)(delayed(fonctions.Reconnaisance)(2,histogrammes_train ,histogrammes_test,k_nn,label_train,label_test,exel_line,nom_file_excel) for k_nn in range(1, 9, 2))#convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante
    '''os.system( 'python LTP.py')
    os.system('python ltp_acp.py')
    os.system('python ALTP.py')
    os.system('python Altp_acp.py')
    os.system('python UniformLBP.py')
    os.system('python CS_LTP.py')






    for k_nn in range(1, 9, 2):
        fonctions.Reconnaisance(3,histogrammes_train ,histogrammes_test,k_nn,label_train,label_test,exel_line,nom_file_excel) #fonction de la reconnaissance k-nn 1,3,5,7'''

