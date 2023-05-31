# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:52:00 2023

@author: caffe
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.random.set_seed(0)
row_num = 35000
test_num= 4000
dos_data = pd.read_csv("C:\CAN_dataset\data\DoS_dataset.csv",nrows=row_num,header=None)
col_names = ['time_stamp','id', 'dlc','d0','d1','d2','d3','d4','d5','d6','d7','R']
dos_data = dos_data.dropna()
dos_data.columns = col_names
dos_data = dos_data[dos_data.R!= 'R']
dos_data.head()

dos_data.tail()
dos_data = dos_data.drop(['time_stamp', 'dlc','R'], axis=1)
#Create features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
def create_feature(data_frame,label,n=30,m=9):
    feature_list = []
    nrow = data_frame.shape[0]
    scale_colums = ['id', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    for col  in dos_data.columns:
        data_frame[col] = data_frame[col].apply(int, base=16)
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
from PIL import Image
from IPython.display import display
img = Image.fromarray(dos_feature_df.features[0], 'L')
display(img)
#for fuzzy dataset
fuzzy_data = pd.read_csv("C:\CAN_dataset\data\Fuzzy_dataset.csv",nrows=row_num,sep=',',header=None)
fuzzy_data.columns = col_names
fuzzy_data =fuzzy_data[fuzzy_data.R!= 'R']
fuzzy_data = fuzzy_data.dropna()
fuzzy_data = fuzzy_data.drop(['time_stamp', 'dlc','R'], axis=1)
fuzzy_feature_df = create_feature(fuzzy_data,"fuzzy")
fuzzy_feature_df.head()
#for gear dataset
gear_data = pd.read_csv("C:\CAN_dataset\data\gear_dataset.csv",nrows=row_num,sep=',',header=None)
gear_data.columns = col_names
gear_data = gear_data.dropna()
gear_data = gear_data.drop(['time_stamp', 'dlc','R'], axis=1)
gear_feature_df = create_feature(gear_data,"gear")
gear_feature_df.head()
#for rpm dataset
rpm_data = pd.read_csv("C:\CAN_dataset\data\RPM_dataset.csv",nrows=row_num,sep=',',header=None)
rpm_data.columns = col_names
rpm_data = rpm_data.dropna()
rpm_data = rpm_data.drop(['time_stamp', 'dlc','R'], axis=1)
rpm_feature_df = create_feature(rpm_data,"rpm")
rpm_feature_df.head()
#normaldata
#norm_data = pd.read_csv(r"C:\Users\caffe\OneDrive\Desktop\normal_run_data.csv", nrows=row_num, sep=',', header=None)
#col_names_norm = ['time_stamp','id', 'dlc','d0','d1','d2','d3','d4','d5','d6','d7']
#norm_data.columns = col_names_norm
#norm_data = norm_data.dropna()
#norm_data = norm_data.drop(['time_stamp', 'dlc'], axis=1)
#norm_data_df = create_feature(norm_data, "norm")
#norm_data_df.head()
#concat data
final_data = pd.concat([dos_feature_df,fuzzy_feature_df,gear_feature_df,rpm_feature_df],ignore_index = True)
final_data.head()
final_data.tail()
from sklearn.utils import shuffle
final_data = shuffle(final_data)
final_data.head()
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(final_data.label)
# transfomed_label = transfomed_label.astype(np.float32)
a = dos_feature_df.features[0]
a.shape
from tensorflow.keras.layers import Input,Conv1D,Dropout,MaxPooling1D,Flatten,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import concatenate
n_time_steps = 30
n_features = 9
n_epoch = 150
(n_features,n_time_steps)

model = Sequential([
    Conv1D(input_shape=(n_time_steps, n_features), filters=16, kernel_size=3, activation='relu'),
    Conv1D(filters=16, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=16),
    Flatten(),
    Dense(8, activation='relu'),
   # Dropout(0.2),  # Add dropout with a dropout rate of 0.2
    Dense(4, activation='softmax')
])


model.compile(optimizer='adam',

          loss='categorical_crossentropy',

          metrics=['categorical_accuracy'])
model.summary()
features = np.concatenate(final_data.features.values)
features = features.reshape(-1,n_time_steps,n_features)
features.shape
#features_train, features_test, labels_train, labels_test = train_test_split(features, transfomed_label, test_size=0.3, random_state=0)
features_train, features_test, labels_train, labels_test = train_test_split(features, transfomed_label, test_size=0.2, random_state=0)

history = model.fit(features_train, labels_train, epochs=n_epoch, batch_size=64, validation_data=(features_test, labels_test))


test_loss, test_accuracy = model.evaluate(features_test, labels_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Calculate predictions
predictions = model.predict(features_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(labels_test, axis=1)

# Calculate confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion_mat)
test_loss, test_accuracy = model.evaluate(features_test, labels_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

val_acc = history.history['val_categorical_accuracy']
import matplotlib.pyplot as plt

# Plot the validation accuracy
epochs = range(1, n_epoch+1)
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
import matplotlib.pyplot as plt
x = list(range(n_epoch))
plt.plot(x, history.history['loss'],label="loss")
plt.plot(x, history.history['val_loss'],label="val_loss")
plt.legend();
plt.plot(x, history.history['val_categorical_accuracy'],label="val_categorical_accuracy")
plt.plot(x, history.history['categorical_accuracy'],label="categorical_accuracy")
plt.legend();