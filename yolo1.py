# -*- coding: utf-8 -*-
# @Author: User
# @Date:   2019-10-18 14:22:58
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-31 11:54:04
import cv2
import os
import time
import numpy as np
import glob
def yolo(raw_img):
    #filename = '6.pgm'

    model = 'yolov3-face'
    scale = 1

    IMG_WIDTH, IMG_HEIGHT = 416, 416
    CONFIDENCE = 0.5
    THRESH = 0.3

    net = cv2.dnn.readNetFromDarknet("Yolo/yolo_models/yolov3-face.cfg", "Yolo/yolo_weights/yolov3-face.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    #raw_img = cv2.imread(os.path.join('TestImg', filename))
    h, w, _ = raw_img.shape
    # img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # inference
    t0 = time.time()

    blob = cv2.dnn.blobFromImage(raw_img, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layers_names = net.getLayerNames()
    outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # only face
            if confidence > CONFIDENCE and class_id == 0:
                box = detection[0:4] * np.array([w, h, w, h])
                centerX, centerY, bwidth, bheight = box.astype('int')
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESH)

    t1 = time.time()
    #print(f"took {round(t1 - t0, 3)} to get {len(idxs.flatten())} faces")

    if boxes is None or confidences is None or idxs is None or class_ids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
    x=0
    y=0


    try:
        x, y = boxes[0][0], boxes[0][1]
        w, h = boxes[0][2], boxes[0][3]
    except :
        print('erreur')


        # cv2.rectangle(raw_img, (x,y), (x+w,y+h), (80,18,236), 2)

    if x <0 :
        x=0
    if y < 0:
        y= 0
    if h < 0:
        h= 0
    if w < 0:
        w = 0
    img1 = raw_img[y:y + h, x:x + w:]
    #cv2.imshow('IMG', img1)
    #cv2.imwrite('savedImage.jpg', img1)
    #cv2.waitKey(0)

    '''font = cv2.FONT_HERSHEY_DUPLEX
    text = f'took {round(t1-t0, 3)} to get {len(idxs.flatten())} faces'
    cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)
    # raw_img = draw_labels_and_boxes(raw_img, boxes, confidences, classids, idxs, colors, labels)
    cv2.imshow('IMG', raw_img)
    cv2.waitKey(0)'''
    return img1

def images_save(path):
    nomder = len(os.listdir(path ))
    print(nomder)
    total=0
    for i in range(0,nomder+1):
        n=str(i)
        nombimg = len(glob.glob1(path + '/' + n + '/', '*'))
        total=total+nombimg
    for ImageName in os.listdir(path + '/' + '1'):
        Image_path = os.path.join(path + '/' + '1', ImageName)
        img = cv2.imread(Image_path, 0)
        break

    ix=0
    print(total, np.shape(img)[0], np.shape(img)[1])
    #data = np.zeros((total, np.shape(img)[0]* np.shape(img)[1]))
    #data.nbytes
    #print(data.nbytes)
    for name in range(1, nomder+1):
        name = str(name)
        os.makedirs('D:\datasets\yolo\\yalB_test\\'+name)
        print('import directory ' + name)
        for ImageName in os.listdir(path+'/' + name):
            Image_path = os.path.join(path+'/'+ name, ImageName)
            img = cv2.imread(Image_path)
            img1 = yolo(img)

            print('D:\datasets\yolo\\yalB_test\\'+name+'\\'+ImageName[:-4]+'.jpg')


            cv2.imwrite('D:\datasets\yolo\\yalB_test\\' + name + '\\' + ImageName[:-4] + '.jpg', img1)







    return
if __name__ == '__main__':
    path='D:\datasets\\yalB_test'
    images_save(path)
