from datetime import datetime, timedelta

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from models import *
import sklearn
from abc import ABC, abstractmethod


random_seed = 33

class vae_model(ABC):

    def __init__(self, n_filters, kernel_size, learning_rate,
               sequence_length, n_features):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        self.latent_dim = 2
        self.sequence_length = sequence_length
        self.n_features = n_features
    

        if self.kernel_size == 3:
            self.nn_dim = 21
        elif self.kernel_size == 5:
            self.nn_dim = 18
        else:
            self.kernel_size = 3
            self.nn_dim = 21

    def set_seed(self, seed):
    
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    def sampling(self, args):
      
        latent_dim = 2
        z_mean, z_log_sigma = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1)

        return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


    def vae_loss(self, inp, mask, out, z_log_sigma, z_mean):
        masked_input = tf.math.multiply(inp, mask)
        masked_output = tf.math.multiply(out, mask)

        #mse = np.sum(np.square(np.subtract(masked_output, masked_input))) / np.sum(mask)
        mse = K.sum(K.square(masked_output - masked_input)) / K.sum(mask)

        reconstruction = mse * self.sequence_length
        kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))

        return reconstruction + kl

    @abstractmethod
    def get_model(self):
        pass
   

class cnn_vae(vae_model):

    def get_model(self):
  

        self.set_seed(random_seed)

        # encoder

        inp = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        mask = tf.keras.Input(shape=(self.sequence_length, self.n_features))


        conv = tf.keras.layers.Conv1D(filters = self.n_filters, kernel_size = self.kernel_size, activation='relu')(inp)
        print(conv.shape)

        max_pool = tf.keras.layers.MaxPool1D(pool_size = 2)(conv) 

        conv = tf.keras.layers.Conv1D(filters = self.n_filters/2, kernel_size = self.kernel_size, activation='relu')(max_pool)
        print(conv.shape)

        enc = tf.keras.layers.Flatten()(conv)

        enc = tf.keras.layers.Dense(self.nn_dim*8, activation="relu")(enc)

        enc = tf.keras.layers.Dense(self.nn_dim*4, activation="relu")(enc)

        enc = tf.keras.layers.Dense(self.nn_dim*2, activation="relu")(enc)

        z = tf.keras.layers.Dense(self.nn_dim, activation="relu")(enc)

        z_mean = tf.keras.layers.Dense(self.latent_dim)(z)
        z_log_sigma = tf.keras.layers.Dense(self.latent_dim)(z)

        encoder = tf.keras.Model([inp], [z_mean, z_log_sigma])

        # decoder

        inp_z = tf.keras.Input(shape=(self.latent_dim,))

        dec = tf.keras.layers.Dense(self.nn_dim)(inp_z)

        dec = tf.keras.layers.Dense(self.nn_dim*2)(dec)

        dec = tf.keras.layers.Dense(self.nn_dim*4)(dec)

        dec = tf.keras.layers.Dense(self.nn_dim*8)(dec)

        dec = tf.keras.layers.Reshape((self.nn_dim, 8))(dec)

        deconv = tf.keras.layers.Conv1DTranspose(filters=self.n_filters/2, kernel_size=self.kernel_size)(dec)
        print(deconv.shape)

        upsample = tf.keras.layers.UpSampling1D(2)(deconv)

        deconv = tf.keras.layers.Conv1DTranspose(filters=self.n_features, kernel_size=self.kernel_size)(upsample)
        print(deconv.shape)


        out = deconv


        decoder = tf.keras.Model([inp_z], out) 

        # encoder and decoder 

        z_mean, z_log_sigma = encoder([inp])
        z = tf.keras.layers.Lambda(self.sampling)([z_mean, z_log_sigma])
        pred = decoder([z])

        vae = tf.keras.Model([inp,  mask], pred)
        vae.add_loss(self.vae_loss(inp, mask, pred, z_log_sigma, z_mean))
        vae.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        return vae

class lstm_vae(vae_model):
    
    def get_model(self):
      
        self.set_seed(random_seed)

        # encoder

        inp = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        mask = tf.keras.Input(shape=(self.sequence_length, self.n_features))


        enc = tf.keras.layers.LSTM(192, input_shape=(self.sequence_length, self.n_features))(inp)

        z = tf.keras.layers.Dense(96, activation="relu")(enc)

        z_mean = tf.keras.layers.Dense(self.latent_dim)(z)
        z_log_sigma = tf.keras.layers.Dense(self.latent_dim)(z)

        encoder = tf.keras.Model([inp], [z_mean, z_log_sigma])

        # decoder

        inp_z = tf.keras.Input(shape=(self.latent_dim,))

        dec = tf.keras.layers.RepeatVector(self.sequence_length)(inp_z)

        dec = tf.keras.layers.LSTM(192, input_shape=(self.sequence_length, self.n_features), return_sequences=True)(dec)

        out = tf.keras.layers.TimeDistributed(Dense(self.n_features))(dec)

        decoder = tf.keras.Model([inp_z], out) 

        # encoder and decoder 

        z_mean, z_log_sigma = encoder([inp])
        z = tf.keras.layers.Lambda(self.sampling)([z_mean, z_log_sigma])
        pred = decoder([z])

        vae = tf.keras.Model([inp,  mask], pred)
        vae.add_loss(self.vae_loss(inp, mask, pred, z_log_sigma, z_mean))
        vae.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        return vae


# LSTM autoencoder

class lstm_ae():
    
    def __init__(self, learning_rate, sequence_length, n_features):
        
        
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.n_features = n_features
        
    
    def ae_loss(self, inp, output, mask):
        initializer = tf.keras.initializers.Ones()
        ones = initializer(shape=(self.sequence_length, self.n_features))

        loss = -1*((inp*K.log(output)) + ((ones - inp) * K.log(ones - output)))

        return loss
    

    def get_lstm_ae(self):


        # Encoding   

        inp = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        print(inp.shape)

    
        encoded = tf.keras.layers.LSTM(48, return_sequences=True)(inp)
        encoded = tf.keras.layers.LSTM(32, return_sequences=True)(encoded)
        # decoder
        decoded = tf.keras.layers.LSTM(48, return_sequences=True)(encoded)

        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_features, activation='sigmoid'))(decoded)

        lstm_ae = tf.keras.Model(inputs=inp, outputs=decoded)

        #lstm_ae.add_loss(self.ae_loss(inp, decoded, mask))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        lstm_ae.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))

        return lstm_ae

   