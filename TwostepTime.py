# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:32:17 2024

@author: caffe
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
initial_time = time.time()
#test_row_num=500
test_data = pd.read_csv(r"C:\",nrows=test_row_num)
test_data = test_data.dropna()

#Variables for training
cols = list(test_data)[0:9]
outcol=list(test_data)[9:10]
df_for_test = test_data[cols].astype(float)
df_output=test_data[outcol].astype(float)

scaler = StandardScaler()
scaler = scaler.fit(df_for_test)
df_for_testing_scaled = scaler.transform(df_for_test)
scaler = scaler.fit(df_output)
df_output_scaled=scaler.transform(df_output)
testX = []
testY = []


#************getting the input shape ready*************
#1 input unit= 10*9 matrix, 1 output unit=10*1 matrix
n_past=5
for i in range(n_past, len(df_for_test)):
    testX.append(df_for_testing_scaled[i - n_past:i, 0:df_for_testing_scaled.shape[1]])
    testY.append(df_output_scaled[i - n_past:i, 0:df_output_scaled.shape[1]])

X_test = np.array(testX)
Y_test=np.array(testY)
print("Testing time 1:", time.time() - initial_time)

loaded_model = load_model("C:\\iic\lstmtriallarge.h5")
print("Testing time 2:", time.time() - initial_time)

Y_predict = loaded_model.predict(X_test)
print("Testing time 3:", time.time() - initial_time)



#############only in case of attack dataset####################



#make inverse transform on Y_predict
inverse_ypred = scaler.inverse_transform(Y_predict)
Y_pred_transformed=np.array(inverse_ypred)
#print(Y_test_transformed)
P,Q= Y_pred_transformed.shape
#print(Y_test_transformed.shape)
Y_1D = Y_pred_transformed.reshape(P*Q,1 )
#created an 1D array for setting threshold

#setting a threshold of 0.5 on invverse_transformed output
Y_1D[Y_1D > 0.5] = 1
Y_1D[Y_1D <0.5] = 0

a,b,c=Y_test.shape
Y_test_2d=Y_test.reshape(a,b)
#taking inverse transform of y_test
inverse_y_test = scaler.inverse_transform(Y_test_2d)
y_test_transformed=np.array(inverse_y_test)
P,Q= y_test_transformed.shape
#converting Y_test to 1D array
Y_test_1D = y_test_transformed.reshape(P*Q,1 )
#making amends for slight errors while making inverse transforms
Y_test_1D[Y_test_1D > 0.5] = 1
Y_test_1D[Y_test_1D < 0.5] = 0

d,e=Y_test_1D.shape
f=int(d/5)

arr=Y_1D.reshape(5,f)
arr_new=np.zeros(f+5)
#print(arr_new)
#getting the actual array of pridicted values for one to one testing##
for j in range(0, np.shape(arr)[1]-1):
    for i in range(0,5):
      k=i+j
      arr_new[k+1]=arr[i][j]+arr_new[k+1]

#####seting threshold2####
for i in range(0,np.shape(arr_new)[0]-1):
    if arr_new[i]>4:
      arr_new[i]=1
    else:
      arr_new[i]=0


arr2=Y_test_1D.reshape(5,f)
#arr2->Y_2D
arr_new2=np.zeros(f+5)
#print(arr_new2)
for j in range(0, np.shape(arr2)[1]-1):
    for i in range(0,5):
      k=i+j
      arr_new2[k+1]=arr2[i][j]+arr_new2[k+1]
####setting threshold2#####
for i in range(0,np.shape(arr_new2)[0]-1):
    if arr_new2[i]>4:
      arr_new2[i]=1
    else:
      arr_new2[i]=0


#print(np.sum(arr_new2))

predicted_y=arr_new2.reshape(-1, 1)
z = np.concatenate((df_for_test, predicted_y), axis=1)
n = df_for_test.shape[0] // 4



dos_data = df_for_test.iloc[0:n, :]
dos_data = dos_data[dos_data.iloc[:, -1] != 0]
fuzzy_data = df_for_test.iloc[n:2*n, :]
fuzzy_data = fuzzy_data[fuzzy_data.iloc[:, -1] != 0]
gear_data = df_for_test.iloc[2*n:3*n, :]
gear_data = gear_data[gear_data.iloc[:, -1] != 0]
rpm_data = df_for_test.iloc[3*n:4*n, :]
rpm_data = rpm_data[rpm_data.iloc[:, -1] != 0]

#Create features

scaler = MinMaxScaler()
def create_feature(data_frame,label,n=30,m=9):
    feature_list = []
    nrow = data_frame.shape[0]
    scale_colums = ['ID(dec)', 'dec1', 'dec2', 'dec3', 'dec4', 'dec5', 'dec6', 'dec7', 'dec8']
    for col  in dos_data.columns:
        data_frame[col] = data_frame[col]
    data_frame[scale_colums] = scaler.fit_transform(data_frame[scale_colums])    
    for i in range(0,nrow,n):
        if nrow >=  i+n:
            tem_file = data_frame.iloc[i:i+n,:].values
            feature_list.append(tem_file)
    feature_df = pd.DataFrame(data={"features":feature_list,"label":[label]*len(feature_list)})
    return feature_df


dos_feature_df = create_feature(dos_data,"dos")
dos_feature_df.shape
dos_feature_df.shape[1]
dos_feature_df.head()

#for fuzzy dataset
fuzzy_feature_df = create_feature(fuzzy_data,"fuzzy")

print(fuzzy_feature_df.head())
#for gear dataset
gear_feature_df = create_feature(gear_data,"gear")

print(gear_feature_df.head())
#for rpm dataset
rpm_feature_df = create_feature(rpm_data,"rpm")
print(rpm_feature_df.head())

final_data = pd.concat([dos_feature_df,fuzzy_feature_df,gear_feature_df,rpm_feature_df],ignore_index = True)
print(final_data.head())
final_data.tail()

final_data = shuffle(final_data)
final_data.head()
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(final_data.label)

n_time_steps = 30
n_features = 9
n_epoch = 150
(n_features,n_time_steps)


features = np.concatenate(final_data.features.values)
features = features.reshape(-1,n_time_steps,n_features)
features.shape
print("Testing time 1:", time.time() - initial_time)

loaded_model = load_model("C:\\iic\cnntwostepmodel.h5")

y_pred = loaded_model.predict(features)


