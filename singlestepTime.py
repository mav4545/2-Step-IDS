import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
#from memory_profiler import profile
import psutil
from sklearn.preprocessing import MinMaxScaler



initial_time = time.time()
df_for_test = pd.read_csv(r"C:\DEC_files\dec2.csv", usecols=range(10), )

n = df_for_test.shape[0] // 4

dos_data_full = df_for_test.iloc[0:n, :]
dos_data = dos_data_full[dos_data_full.iloc[:, -1] != 0]
dos_data = dos_data.iloc[:, :-1]
fuzzy_data_full = df_for_test.iloc[n:2*n, :]
fuzzy_data= fuzzy_data_full[fuzzy_data_full.iloc[:, -1] != 0]
fuzzy_data = fuzzy_data.iloc[:, :-1]
gear_data_full = df_for_test.iloc[2*n:3*n, :]
gear_data = gear_data_full[gear_data_full.iloc[:, -1] != 0]
gear_data = gear_data.iloc[:, :-1]
rpm_data_full = df_for_test.iloc[3*n:4*n, :]
rpm_data = rpm_data_full[rpm_data_full.iloc[:, -1] != 0]
rpm_data = rpm_data.iloc[:, :-1]
norm_data = pd.read_csv(r"C:\DEC_files\dec2.csv", usecols=range(10), nrows=None)
norm_data = norm_data[norm_data.iloc[:, 9] == 0]
norm_data = norm_data.iloc[:, :-1]



scaler = MinMaxScaler()
def create_feature(data_frame,label,n=30,m=9):
    feature_list = []
    nrow = data_frame.shape[0]
    scale_colums = ['ID(dec)', 'dec1', 'dec2', 'dec3', 'dec4', 'dec5', 'dec6', 'dec7', 'dec8']
    data_frame[scale_colums] = scaler.fit_transform(data_frame[scale_colums])    
    for i in range(0,nrow,n):
        if nrow >=  i+n:
            tem_file = data_frame.iloc[i:i+n,:].values
            feature_list.append(tem_file)
    feature_df = pd.DataFrame(data={"features":feature_list,"label":[label]*len(feature_list)})
    return feature_df
if dos_data.shape[0] > 1:
    # Assuming create_feature is a function that processes fuzzy_data
    dos_feature_df = create_feature(dos_data,"dos")    
    dos_feature_df.shape
else:
    dos_feature_df = None
    
if fuzzy_data.shape[0] > 1:
    # Assuming create_feature is a function that processes fuzzy_data
    fuzzy_feature_df = create_feature(fuzzy_data, "fuzzy")
    print(fuzzy_feature_df.head())

else:
    fuzzy_feature_df = None

if gear_data.shape[0] > 1:
    # Assuming create_feature is a function that processes fuzzy_data
    gear_feature_df = create_feature(gear_data,"gear")
    gear_feature_df.head()
else:
    gear_feature_df = None

if rpm_data.shape[0] > 1:
    # Assuming create_feature is a function that processes fuzzy_data
    rpm_feature_df = create_feature(rpm_data,"rpm")
    rpm_feature_df.head()
else:
    rpm_feature_df = None

#normaldata
if norm_data.shape[0] > 1:
    # Assuming create_feature is a function that processes fuzzy_data
    norm_data_df = create_feature(norm_data, "norm")
    norm_data_df.head()
else:
    norm_data_df = None
    

##concat data
final_data = pd.concat([dos_feature_df,fuzzy_feature_df,gear_feature_df,rpm_feature_df,norm_data_df],ignore_index = True)
final_data.head()
final_data.tail()
from sklearn.utils import shuffle
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
#features_train, features_test, labels_train, labels_test = train_test_split(features, transfomed_label, test_size=0.3, random_state=0)

loaded_model =  load_model("C:\\iic\singlestep16.h5")

y_pred = loaded_model.predict(features)
print("Testing time 1:", time.time() - initial_time)
