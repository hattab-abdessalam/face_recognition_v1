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
from sklearn import  preprocessing
def calculate(type,n, path_faces):#calculer les histogrammes ltp du dossier n qui trouve dans le lien path_faces

    derName = str(n)# convertir le nom du dossier en string
    name = str(n)
    i = 0
    t=5 #threshold
    nombimg = len(glob.glob1(path_faces + '/' + derName + '/', '*'))#calculer le nombre des visages dans le dossier derName
    histo = np.zeros((nombimg, 512), dtype='int') #matrice pour sauvegarder les histogrammes lbp du dossier derName

    for ImageName in os.listdir(path_faces + '/' + name):
        Image_path = os.path.join(path_faces + '/' + name, ImageName)
        #print(Image_path)
        img = cv2.imread(Image_path, 0)
        img = cv2.resize(img, (128,128 ))


        matrice_ltp1,matrice_ltp2 = calculate_matrice_LTP(img,t)#calculer matrice LTP de (img )
        h1 = fonctions.histogramme(type,np.array(matrice_ltp1)) #calculer histogramme ltp upper
        h2 = fonctions.histogramme(type,np.array(matrice_ltp2))##calculer histogramme ltp lower
        hi = np.concatenate((h1, h2))
        histo[i, :] = hi #Sauvegarder histogramme dans la matrice des histogrammes
        print(name,'-------->', i) #afficher le nombre des images déjà traité
        i += 1
    return histo




def calculate_matrice_LTP(image,t):
    img_LTP1 = np.zeros((np.shape(image)[0]-2, np.shape(image)[1]-2)) #matrice lTp upper -2 puisque LTP(3*3)
    img_LTP2 = np.zeros((np.shape(image)[0] - 2, np.shape(image)[1] - 2))#matrice ltp lower -2 puisque LTP(3*3)
    for line in range(np.shape(image)[0]-2):
        for column in range(np.shape(image)[1]-2):
            img_LTP1[line, column],img_LTP2[line, column] = calculateLTP(image, column+1,line+1,t)#calculer les codes ltp upper,ltp lower  pour le pixel (line,column)

    return img_LTP1,img_LTP2


def calculateLTP(image, column, line,t):
    neighbours=get_neighbours(image, column, line)
    center=np.array(image)[line, column]
    values1,values2 = calculate_code(center, neighbours,t) #calculer les codes binaire ltp upper ,ltp lower
    # convertir les codes binaire en décimale
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    LTP1 = 0
    LTP2 = 0
    for i in range(0, len(values1)):
        LTP1 += values1[i] * weights[i]
        LTP2 += values2[i] * weights[i]
    return LTP1,LTP2


def get_neighbours(image, column, line):#obtenir les voisins du pixel (column,line)
    bloc=image[line - 1:line - 1+3,column - 1:column - 1+3]
    a= np.array(bloc).flatten()
    neighbours = [a[0],a[1],a[2],a[5],a[8],a[7],a[6],a[3]]
    return neighbours


def calculate_code( center, neighbours,t):#calculer les codes binaire

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
    lTPcode2 =np.copy(result)

    for i in range(0, 8):    #ltp upper (modifier -1 par un 0)
        if lTPcode1[i] ==-1 :
            lTPcode1[i]=0

    for i in range(0, 8): #ltp lower (modifier 1 par un 0 et le -1 par un 1)
        if lTPcode2[i] == 1:
            lTPcode2[i] = 0
        if lTPcode2[i] == -1:
            lTPcode2[i] = 1

    return lTPcode1,lTPcode2



if __name__ == '__main__':
    print('LTP')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)    #afficher l'heure de démarrage


    train_path = "D://datasets//crossvalidation//data//feret_gray//new//feret_gray_5_6//train"  # lien du dossier (train)
    test_path = "D://datasets//crossvalidation//data//feret_gray//new//feret_gray_5_6//test"  # lien du dossier (test)
    xls_path = 'orl_train'
    nbder_train=len(os.listdir(train_path))#nombre des dossiers (personnes) dans le dossier train

    num_cores = multiprocessing.cpu_count()#nombre des coeurs cpu
    print(num_cores)

    histogrammes= Parallel(n_jobs=num_cores)(delayed(calculate)(1,kl, train_path) for kl in tqdm(range(1, nbder_train+1)))#calculer histogrammes ltp (train)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time) #l'heure de fin

    np.save('orl1_histo_ltp_train.npy', histogrammes )#souvgarder les histogrammes dans le disque dur
    histogrammes = np.load('orl1_histo_ltp_train.npy',allow_pickle=True)

    histogrammes_train=fonctions.conv_list_arr(histogrammes,fonctions.nomb_img( train_path))#convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante

    label_train = fonctions.labels(train_path)#calculer labels train

    nbder_test=len(os.listdir(test_path)) # nombre des dossiers (personnes) dans le dossier test
    histogrammes_test= Parallel(n_jobs=num_cores)(delayed(calculate)(1,kl, test_path) for kl in tqdm(range(1,  nbder_test+1))) #convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante

    np.save('orl1_histo_ltp_test.npy', histogrammes_test)#souvgarder les histogrammes dans le disque dur
    histogrammes_test = np.load('orl1_histo_ltp_test.npy',allow_pickle=True)

    histogrammes_test=fonctions.conv_list_arr(histogrammes_test,fonctions.nomb_img( test_path))#convertir  histogrammes_test en 2 dimensions puisque chaque coeur calculer une matrice dépendante

    '''mix = np.concatenate((histogrammes_train, histogrammes_test), axis=0)
    prepare = preprocessing.MaxAbsScaler() #MaxAbsScaler() MinMaxScaler()
    x = prepare.fit_transform(mix.astype(float))
    histogrammes_train = x[0:histogrammes_train.shape[0], :]
    histogrammes_test = x[histogrammes_train.shape[0]:, :]
    print(histogrammes_test.shape)'''
    label_test = fonctions.labels(test_path)#calculer labels test pour vérifer la reconnaissance
    exel_line = 2  # Numéro de la ligne dans le fichier Excel pour écrire les résultats de la reconnaissance
    nom_file_excel = 'knn_' + xls_path[:-6] + '.xlsx'  # nom du fichier Excel qui est utilisé pour écrire les résultats de la reconnaissance

    #Parallel(n_jobs=num_cores)(delayed(fonctions_yal.Reconnaisance)(histogrammes_train ,histogrammes_test,k_nn,label_train,label_test,exel_line,nom_file_excel) for k_nn in range(1, 9, 2))#convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante
    Parallel(n_jobs=num_cores)(delayed(fonctions.Reconnaisance)(2,histogrammes_train ,histogrammes_test,k_nn,label_train,label_test,exel_line,nom_file_excel) for k_nn in range(1, 9, 2))#convertir  histogrammes en 2 dimensions puisque chaque coeur calculer une matrice dépendante

    '''for k_nn in range(1, 9, 2):
        fonctions.Reconnaisance(3, histogrammes_train ,histogrammes_test,k_nn,label_train,label_test,exel_line,nom_file_excel) #fonction de la reconnaissance k-nn 1,3,5,7'''
