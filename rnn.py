# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling --- We apply normalization [(x-min)/(max-min)]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# 60 time steps means at each time t the rnn is going to look at the 60 stock prices before time t i.e. 
# the stock prices between 60 days before t and t and based on the trends it capture it will predict the 1 output
# each month has 20 financial days. so 60 days means 3 months. we look back 3 months everytime

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
# adding number of predictors (indicators)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    



# Part 2 - Building the RNN

# Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN 
regressor = Sequential()


# Part 3 - Make predictions and visualization
 