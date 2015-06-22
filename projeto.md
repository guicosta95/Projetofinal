# -*- coding: utf-8 -*-
import numpy as np
import cv2
import cv
  
ESC=27 #codigo Ascii para a tecla esc  
camera = cv2.VideoCapture(1)#pega o feed da camera do computador
orb = cv2.ORB()# cria uma instancia ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)# usa o brute force keypoint descriptor com o metodo hamming

imgoriginalColor=cv2.imread('modeloinsper.jpg')#lendo a imagem modelo
imgTrainGray = cv2.cvtColor(imgoriginalColor, cv2.COLOR_BGR2GRAY)# coloca a imagem em branco e preto

kpTrain = orb.detect(imgTrainGray,None)# pega os keypoints da imagem original
kpTrain, desTrain = orb.compute(imgTrainGray, kpTrain)# computa todos os descriptors dos keypoints detectados

firsttime=True

while True:
   
    ret, imgCamColor = camera.read() #lendo as informaçoes da camera
    imgCamGray = cv2.cvtColor(imgCamColor, cv2.COLOR_BGR2GRAY)# transforma em grayscale
    kpCam = orb.detect(imgCamGray,None)# pega os keypoints da imagem da camera
    kpCam, desCam = orb.compute(imgCamGray, kpCam)# computa todos os descriptors dos keypoints detectados
    matches = bf.match(desCam,desTrain)# aplica o matching entre os keypoints das duas imagens
    dist = [m.distance for m in matches]# coloca um threshold entre a distancia dos arrays para detectar os erros 
    thres_dist = (sum(dist) / len(dist)) * 0.5
    matches = [m for m in matches if m.distance < thres_dist]# eliminando os erros baseados no threshold acima

    if firsttime==True: # coloca um tamanho para as imagens sendo comparadas para depois serem mostradas para o usuario
        h1, w1 = imgCamColor.shape[:2]
        h2, w2 = imgoriginalColor.shape[:2]
        nWidth = w1+w2 
        nHeight = max(h1, h2)
        hdif = (h1-h2)/2
        firsttime=False
       
    result = np.zeros((nHeight, nWidth, 3), np.uint8)
    result[hdif:hdif+h2, :w2] = imgoriginalColor # pega os resultados das duas imagens em cores
    result[:h1, w2:w1+w2] = imgCamColor

    for i in range(len(matches)):
        pt_a=(int(kpTrain[matches[i].trainIdx].pt[0]), int(kpTrain[matches[i].trainIdx].pt[1]+hdif)) # cria linhas pra cada matching point achado pelo programa 
        pt_b=(int(kpCam[matches[i].queryIdx].pt[0]+w2), int(kpCam[matches[i].queryIdx].pt[1]))
        cv2.line(result, pt_a, pt_b, (255, 0, 0))

    cv2.imshow('Camara', result)# mostra a umagem da camera e da imagem original
    print(len(matches))# imprime o numero de matches entre a camera e a imagem original
    if len(matches)>20: # checam se a carteirinha é da faculdade a partir do numero de matches
        print("desconto concedido")
        break
    else:
        print("carteira nao reconhecida continue tentando")
   
    
    key = cv2.waitKey(20)                             
    if key == ESC:# espera a tecla esc ser apertada para fechar o programa
        break

cv2.destroyAllWindows()
camera.release()
