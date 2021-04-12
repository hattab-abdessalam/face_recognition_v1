import numpy as np
import cv2 as cv
import os
import glob


def compare(lis, i, j):
    a = np.array(lis)
    b = 0
    if a.shape[0] != 0:
        for k in range(0, a.shape[0]):
            if a[k][0] > i + 15 or a[k][1] > j + 15 or a[k][0] < i - 15 or a[k][1] < j - 15:
                b = b + 1
    if b == a.shape[0]:
        return True
    else:
        return False


harris = 'harris_orl_train\\'
path = 'd://datasets//orl_train'
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
    img = cv.imread(Image_path, 0)
    break
ix = 0
for name in range(1, nomder + 1):
    name = str(name)
    print('import directory ' + name)
    os.mkdir(harris + name)

    for ImageName in os.listdir(path + '/' + name):
        A = 0
        B = 0
        im = 0
        Image_path = os.path.join(path + '/' + name, ImageName)
        lis = []
        img = cv.imread(Image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        gray = np.float32(gray)
        # Detecting corners
        dst = cv.cornerHarris(gray, 2, 3, 0.04)  # Normalizing
        print(dst)
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        dst_norm_scaled = cv.convertScaleAbs(dst_norm)
        # Drawing a circle around corners

        os.mkdir(harris + name + '\\' + ImageName[:-4])
        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):

                if int(dst_norm[i, j]) > 130:

                    if compare(lis, i, j):
                        a = i
                        b = j
                        print(a, b)
                        if i < 16:
                            a = 16
                        if j < 16:
                            b = 16
                        if i > img.shape[0] - 16:
                            a = img.shape[0] - 16
                        if j > img.shape[1] - 16:
                            b = img.shape[1] - 16

                        cv.imwrite(harris + name + '\\' + ImageName[:-4] + '\\' + str(im) + '.jpg',
                                   img[a - 16:a + 16, b - 16:b + 16])
                        im = im + 1
                        lis.append([i, j])
'''  
if i>16 and  i<img.shape[0]-16 and j>16  and j<img.shape[1]-16:
                            im=im+1
                            #cv.imshow('corners_window', img[i-16:i+16,j-16:j+16])
                            cv.imwrite('harris_orl_test\\'+name+'\\'+ImageName[:-4]+'\\'+str(im)+'.jpg' , img[i-16:i+16,j-16:j+16])
                            lis.append([i,j])
                        else:
                            if i<16 and j <16 :
                                im=im+1
                                cv.imwrite('harris_orl_test\\'+name+'\\'+ImageName[:-4]+'\\'+str(im)+'.jpg' , img[0:32,0:32])
                                lis.append([i,j])
                                #cv.imshow('corners_window', img[0:32,0:32])  
                            else:
                                 if i<16 and j<img.shape[1]-16 :
                                     im=im+1
                                     cv.imwrite('harris_orl_test\\'+name+'\\'+ImageName[:-4]+'\\'+str(im)+'.jpg' , img[0:32,j-16:j+16])
                                     lis.append([i,j])
                                     #cv.imshow('corners_window', img[0:32,j-16:j+16])
                                 if j<16 and i<img.shape[0]-16 : 
                                     im=im+1
                                     cv.imwrite('harris_orl_test\\'+name+'\\'+ImageName[:-4]+'\\'+str(im)+'.jpg' , img[i-16:i+16,0:32])

                                     lis.append([i,j])
                                     #cv.imshow('corners_window', img[i-16:i+16,0:32])
                                 if i>16 and j>img.shape[1]-16 :
                                     im=im+1
                                     cv.imwrite('harris_orl_test\\'+name+'\\'+ImageName[:-4]+'\\'+str(im)+'.jpg' , img[i-16:i+16,img.shape[1]-32:img.shape[1]])
                                     lis.append([i,j])
                                     #cv.imshow('corners_window', img[0:32,j-16:j+16])
                                 if j>16 and i>img.shape[0]-16 : 
                                     im=im+1
                                     cv.imwrite('harris_orl_test\\'+name+'\\'+ImageName[:-4]+'\\'+str(im)+'.jpg' , img[img.shape[0]-32:img.shape[0],j-16:j+16])

                                     lis.append([i,j])
                                     #cv.imshow('corners_window', img[i-16:i+16,0:32])

                                 if  i>img.shape[0]-16 and j>img.shape[1]-16: 
                                     im=im+1
                                     cv.imwrite('harris_orl_test\\'+name+'\\'+ImageName[:-4]+'\\'+str(im)+'.jpg' , img[img.shape[0]-32:j-16:j+16])

                                     lis.append([i,j])
                                     #cv.imshow('corners_window', img[i-16:i+16,0:32])


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
# Detecting corners
dst = cv.cornerHarris(gray, 2, 3, 0.04)# Normalizing
dst_norm = np.empty(dst.shape, dtype=np.float32)
cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
dst_norm_scaled = cv.convertScaleAbs(dst_norm)
# Drawing a circle around corners
A=0
B=0
for i in range(dst_norm.shape[0]):
    for j in range(dst_norm.shape[1]):
        if int(dst_norm[i, j]) > 130:
            if (i>A+20 or j>B+20 or i<A-20 or j<B-20):
                print(i>A+20 , j>B+20 , i<A-20 , j<B-20)
                print(i,j)
                cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
                if i>14 and j>15 :
                    #cv.imshow('corners_window', img[i-16:i+16,j-16:j+16])
                    cv2.imwrite(filename, image)
                else:
                    if i<15 and j <15:
                        #cv.imshow('corners_window', img[0:32,0:32])  
                    else:
                         if i<14 :
                             #cv.imshow('corners_window', img[0:32,j-16:j+16])
                         else :
                             #cv.imshow('corners_window', img[i-16:i+16,0:32])



                cv.waitKey(0)
                A=i
                B=j
cv.imshow('corners_window', dst_norm_scaled)
dst = cv.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('dst', img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()'''
