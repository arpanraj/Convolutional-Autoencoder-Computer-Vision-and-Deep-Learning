#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 19:07:15 2022

@author: arpanrajpurohit
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

dataset = tf.keras.datasets.fashion_mnist
(train_input, train_output), (test_input, test_output) = dataset.load_data()

test_input = test_input / 255.0

test_input = np.reshape(test_input, (len(test_input),784))

pca = PCA(n_components = 2)
pca_proc = pca.fit_transform(test_input)
pca_proc = np.append(pca_proc, np.reshape(test_output,(len(test_output),1)), 1)

pca_dataframe = pd.DataFrame(data = pca_proc,
                        columns = ["x",
                                   "y", "out"])

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="x", y="y",
    hue = "out",
    hue_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    palette=sns.color_palette("hls", 10),
    data=pca_dataframe,
    legend="full",
    alpha=0.3
)
