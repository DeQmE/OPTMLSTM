#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Adam Ntakaris (adamantios.ntakaris@ed.ac.uk, @gmail.com)
"""

from keras.layers import Dense
import keras
import numpy as np
import OPTMCell


# Input Dim --> (num_of_data_samples, 1, 41)
# 41 = 40 LOB Price and Volume levels + Current mid_price/Guarantor
three_dim_inpt = np.random.rand(600, 1, 41)

# Regression Labels --> [mid_prices,]
lbls = np.random.rand(600,)

batch_size = 1
num_of_hidden_units = 8

input_1 = keras.Input(batch_shape = (batch_size, 1, 41))
layer_1 = keras.layers.RNN(OPTMCell.RevisedLSTMCell(num_of_hidden_units), 
                           return_sequences=True, stateful=False)(input_1)
output_1 = Dense(1)(layer_1)

model = keras.Model(inputs=input_1, outputs=output_1)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.fit(three_dim_inpt, lbls, batch_size=1, epochs=5)

