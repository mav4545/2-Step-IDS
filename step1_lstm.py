import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


file_path = "C:\DEC_files"
#list all the files from the directory
file_list = os.listdir(file_path)
file_list

file1 = "C:\DEC_files\dec1.csv"
file2 = "C:\DEC_files\dec2.csv"
file3 = "C:\DEC_files\dec3.csv"
file4 = "C:\DEC_files\dec4.csv"
df = pd.concat(
    map(lambda file: pd.read_csv(file, nrows=2500), [file1, file2, file3, file4]),
    ignore_index=True
)

print(df.head)

df.output.value_counts()
#Assuming 1=attack and 0=no attack

#Variables for training
cols = list(df)[0:9]
outcol=list(df)[9:10]
print(outcol)

print(cols)
print(outcol)

df_for_training = df[cols].astype(float)
df_output=df[outcol].astype(float)
print(df_output.shape)
print((df_for_training).shape)
#retain relavant info from csv.

# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)
# normalize the input
scaler = scaler.fit(df_output)
df_output_scaled=scaler.transform(df_output)
# normalize the output
print(df_for_training_scaled.shape)
print(df_output_scaled.shape)

#getting the input shape ready
trainX = []
trainY = []

print(len(df_output))
print(len(df_for_training_scaled))
print(df_output.shape[1])
print(df_for_training.shape[1])
print(df_output.head)

#************getting the input shape ready*************
#1 input unit= 10*9 matrix, 1 output unit=10*1 matrix
n_past=10          
for i in range(n_past, len(df_for_training_scaled)):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_output_scaled[i - n_past:i, 0:df_output.shape[1]])

X, Y = np.array(trainX), np.array(trainY)

#spliting dataset into train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

#LSTM model Design
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1]))

#adam optimizer and compilation
model.compile(optimizer='adam', loss='mse')
model.summary()

print(Y_test)

history = model.fit(X_train, Y_train, epochs=15, batch_size=64, validation_split=0.2,shuffle=False, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

#pridiction/getting Y_predict
current_directory = os.getcwd()
model.save(os.path.join(current_directory, "lstmoptimized.h5"))

Y_predict=model.predict(X_test)
print(Y_predict)

#make inverse transform on Y_predict
inverse_ypred = scaler.inverse_transform(Y_predict)
Y_pred_transformed=np.array(inverse_ypred)
#print(Y_test_transformed)
P,Q= Y_pred_transformed.shape
#print(Y_test_transformed.shape)
Y_1D = Y_pred_transformed.reshape(P*Q,1 )
print(Y_1D.shape)
print(Y_1D)
#created an 1D array for setting threshold

#setting a threshold of 0.5 on invverse_transformed output
Y_1D[Y_1D > 0.5] = 1
Y_1D[Y_1D <0.5] = 0
print(Y_1D)
print(np.sum(Y_1D))


a,b,c=Y_test.shape
print(a,b,c)
Y_test_2d=Y_test.reshape(a,b)
print(Y_test_2d.shape)
#taking inverse transform of y_test
inverse_y_test = scaler.inverse_transform(Y_test_2d)
y_test_transformed=np.array(inverse_y_test)
P,Q= y_test_transformed.shape
#converting Y_test to 1D array
Y_test_1D = y_test_transformed.reshape(P*Q,1 )
#making amends for slight errors while making inverse transforms
Y_test_1D[Y_test_1D > 0.5] = 1
Y_test_1D[Y_test_1D < 0.5] = 0
print(np.sum(Y_test_1D))

print(Y_test_1D.shape)

print(Y_test_1D)

#acurracy metrix
from sklearn.metrics import accuracy_score
accuracy_score(Y_test_1D, Y_1D)

print(Y_1D.shape)
print(Y_test_1D.shape)
d,e=Y_test_1D.shape
f=int(d/10)
print(type(f))
print(d,e,f)


################reshaping Y_test_1d into 2d array with 10 as number of rows#######

arr=Y_1D.reshape(10,f)
print(arr.shape)
arr_new=np.zeros(f+10)
for j in range(0, np.shape(arr)[1]-1):
    for i in range(0,9):
      k=i+j
      arr_new[k+1]=arr[i][j]+arr_new[k+1]
print(arr_new)
print(np.sum(arr_new))
#####seting threshold2####
for i in range(0,np.shape(arr_new)[0]-1):
    if arr_new[i]>4:
      arr_new[i]=1
    else:
      arr_new[i]=0

print(np.sum(arr_new))

arr2=Y_test_1D.reshape(10,f)
#arr2->Y_2D
print(arr2.shape)
arr_new2=np.zeros(f+10)
#print(arr_new2)
for j in range(0, np.shape(arr2)[1]-1):
    for i in range(0,9):
      k=i+j
      arr_new2[k+1]=arr2[i][j]+arr_new2[k+1]
####setting threshold2#####
print(arr_new2)
print(np.sum(arr_new2))
for i in range(0,np.shape(arr_new2)[0]-1):
    if arr_new2[i]>4:
      arr_new2[i]=1
    else:
      arr_new2[i]=0

print(np.sum(arr_new2))

print(arr_new2.shape)
print(arr_new.shape)
count=0
for i in range(0,260164):
    if arr_new2[i]==1:
     count=count+1
    i=i+1
print(count)

print(accuracy_score(arr_new2, arr_new))
from sklearn.metrics import accuracy_score, classification_report
report = classification_report(arr_new2, arr_new)
print(report)