# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 02:52:16 2023

@author: lambe
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

data = pd.read_csv('HistoricalData_1679727358631.csv')
closing_prices = data['Low'].to_numpy().reshape(-1, 1)
num_components = 3
num_dims = 1
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(num_dims,)), tfp.layers.MixtureSameFamily(num_components, tfp.distributions.Normal)])
negloglik = lambda y_true, y_pred: -y_pred.log_prob(y_true)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
history = model.fit(closing_prices, epochs=100, batch_size=32)

x = np.linspace(0, 500, 500)
x = x.reshape(-1, 1)
predicted_probs = model.predict(x)

plt.scatter(closing_prices, np.zeros_like(closing_prices), alpha=0.2)
plt.plot(x, predicted_probs[:, :, 0], 'r-', label='Component 1')
plt.plot(x, predicted_probs[:, :, 1], 'g-', label='Component 2')
plt.plot(x, predicted_probs[:, :, 2], 'b-', label='Component 3')
plt.legend()
plt.show()
