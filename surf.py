# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:08:20 2020

@author: BS
"""
import numpy as np
import cv2
import os
import glob

def compare(lis, i, j):
    a = np.array(lis)
    b = 0
    if a.shape[0] != 0:
        for k in range(0, a.shape[0]):
            if (a[k][0] > i + 10 or a[k][1] < j - 10 or a[k][1] > j +10 or a[k][0] < i - 10) :
                b = b + 1
    if b == a.shape[0]:
        return True
    else:
        return False


surf_path = 'surf_orl_train\\'
path = 'orl_train'
nomder = len(os.listdir(path))
print(nomder)
total = 0
for i in range(1, nomder + 1):
    n = str(i)
    nombimg = len(glob.glob1(path + '/' + n + '/', '*'))
    total = total + nombimg
for ImageName in os.listdir(path + '/' + '1'):
    Image_path = os.path.join(path + '/' + '1', ImageName)
    print(Image_path)
    img = cv2.imread(Image_path, 0)
    break
ix = 0
for name in range(1, nomder + 1):
    name = str(name)
    print('import directory ' + name)
    os.mkdir(surf_path + name)


    for ImageName in os.listdir(path + '/' + name):
        im=0
        Image_path = os.path.join(path+'/'+ name, ImageName)
        lis=[]
        os.mkdir(surf_path + name + '\\' + ImageName[:-4])
        img = cv2.imread(Image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        surf = cv2.xfeatures2d.SURF_create()

        keypoints_surf, descriptors = surf.detectAndCompute(img, None)
        print(keypoints_surf)
        pts = cv2.KeyPoint_convert(keypoints_surf)
        pts = np.array(pts)
        print(pts.shape)

        z=16

        for k in range(0, pts.shape[0]):
            i = int(pts[k][0])
            j = int(pts[k][1])
            a = int(pts[k][0])
            b = int(pts[k][1])
            #if compare(lis, i, j):
            print(i, b)
            if i < z:
                a = z
            if j < z:
                b = z
            if i > img.shape[0] - z:
                a = img.shape[0] - z
            if j > img.shape[1] - z:
                b = img.shape[1] - z
            print(surf_path + str(name) + '\\' + ImageName[:-4] + '\\' + str(im) + '.jpg')

            cv2.imwrite(surf_path + str(name) + '\\' + ImageName[:-4] + '\\' + str(im) + '.jpg',
                        img[a - z:a + z, b - z:b + z])
            im = im + 1

                #lis.append([i, j])







