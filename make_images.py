import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
x = pd.read_csv("fer2013.csv")
data = np.array(x)

testangry = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/training/angry0'
testdisgust = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/training/disgust1'
testfear = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/training/fear2'
testhappy = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/training/happy3'
testsad = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/training/sad4'
testsurprise = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/training/surprise5'
testneutral = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/training/neutral6'

valangry = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/validation/angry0'
valdisgust = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/validation/disgust1'
valfear = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/validation/fear2'
valhappy = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/validation/happy3'
valsad = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/validation/sad4'
valsurprise = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/validation/surprise5'
valneutral = 'C:/Users/Sunny/PycharmProjects/emotiondetection2/venv/validation/neutral6'

testemotions = {
    0 : testangry,
    1 : testdisgust,
    2 : testfear,
    3 : testhappy,
    4 : testsad,
    5 : testsurprise,
    6 : testneutral
}
valemotions = {
    0 : valangry,
    1 : valdisgust,
    2 : valfear,
    3 : valhappy,
    4 : valsad,
    5 : valsurprise,
    6 : valneutral
}
# print(data[0][1])
# print(data[1][1])
print(int(data.shape[0]*.8))
print(int(data.shape[0]*.2))



for j in range(0,int(data.shape[0]*.8)):
    y = [int(i) for i in data[j][1].split()]
    #print(y)
    y = np.reshape(y, (48, 48))
    #print(y.shape)
    print(testemotions[data[j][0]])
    y = y.astype(np.uint8)
    cv2.imwrite(os.path.join(testemotions[data[j][0]], str(j + 1) + '.jpg'), y)
    cv2.waitKey(1)

for j in range(int(data.shape[0]*.8) + 1,int(data.shape[0]*.2) + 1 + int(data.shape[0]*.8)):
    y = [int(i) for i in data[j][1].split()]
    #print(y)
    y = np.reshape(y, (48, 48))
    #print(y.shape)
    print(valemotions[data[j][0]])
    y = y.astype(np.uint8)
    cv2.imwrite(os.path.join(valemotions[data[j][0]], str(j + 1) + '.jpg'), y)
    cv2.waitKey(1)

print('DONE DONE DONE DONE')