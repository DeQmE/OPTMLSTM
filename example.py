#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 02:14:45 2022

@author: adam (adamantios.ntakaris@ed.ac.uk, @gmail.com)

"""

from keras.layers import Dense
import keras
import numpy as np
import RevisedLSTMCell


# Input Dim --> (number of data samples, 1, 41)
# 41 =  40 LOB Price and Volume levels + current mid_price/guarantor  
three_dim_inpt = np.random.rand(600, 1, 41)

# Regression Labels -->  [mid_prices,]
lbls = np.random.rand(600,)

batch_size = 1
num_of_hidden_units = 8
copies = 1 # Curse of Dimensionality Helper within the Layer

input_1 = keras.Input(batch_shape = (batch_size, 1,41))
layer_1 = keras.layers.RNN(RevisedLSTMCell(num_of_hidden_units), return_sequences=True, stateful=False)(input_1)
output_1 = Dense(1)(layer_1)

model = keras.Model(inputs=input_1, outputs=output_1)
model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse']
      )
model.fit(three_dim_inpt, lbls, batch_size=1, epochs=5)
