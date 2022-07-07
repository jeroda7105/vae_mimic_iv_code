
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
import xgboost as xgb


random_seed = 33

def convert_to_nan(first_48_data):
    for idx, row in first_48_data.iterrows():
        
        # Making gcs scores nan where unobserved
        if row['x0_nan'] == 1:
            first_48_data.at[idx, 'x0_0.0'] = np.nan
            first_48_data.at[idx, 'x0_1.0'] = np.nan

        if row['x1_nan'] == 1:
            first_48_data.at[idx, 'x1_None'] = np.nan
            first_48_data.at[idx, 'x1_Spontaneously'] = np.nan
            first_48_data.at[idx, 'x1_To Pain'] = np.nan
            first_48_data.at[idx, 'x1_To Speech'] = np.nan

        if row['x2_nan'] == 1:
            first_48_data.at[idx, 'x2_Abnormal Flexion'] = np.nan
            first_48_data.at[idx, 'x2_Abnormal extension'] = np.nan
            first_48_data.at[idx, 'x2_Flex-withdraws'] = np.nan
            first_48_data.at[idx, 'x2_Localizes Pain'] = np.nan
            first_48_data.at[idx, 'x2_No response'] = np.nan
            first_48_data.at[idx, 'x2_Obeys Commands'] = np.nan

        if row['x3_nan'] == 1:
            first_48_data.at[idx, 'x3_10.0'] = np.nan 
            first_48_data.at[idx, 'x3_11.0'] = np.nan
            first_48_data.at[idx, 'x3_12.0'] = np.nan
            first_48_data.at[idx, 'x3_13.0'] = np.nan
            first_48_data.at[idx, 'x3_14.0'] = np.nan
            first_48_data.at[idx, 'x3_15.0'] = np.nan
            first_48_data.at[idx, 'x3_3.0'] = np.nan
            first_48_data.at[idx, 'x3_4.0'] = np.nan
            first_48_data.at[idx, 'x3_5.0'] = np.nan
            first_48_data.at[idx, 'x3_6.0'] = np.nan
            first_48_data.at[idx, 'x3_7.0'] = np.nan
            first_48_data.at[idx, 'x3_8.0'] = np.nan
            first_48_data.at[idx, 'x3_9.0'] = np.nan


        if row['x4_nan'] == 1:
            first_48_data.at[idx, 'x4_Confused'] = np.nan
            first_48_data.at[idx, 'x4_Inappropriate Words'] = np.nan
            first_48_data.at[idx, 'x4_Incomprehensible sounds'] = np.nan
            first_48_data.at[idx, 'x4_No Response'] = np.nan
            first_48_data.at[idx, 'x4_No Response-ETT'] = np.nan
            first_48_data.at[idx, 'x4_Oriented'] = np.nan

    return first_48_data
            



def create_time_series_data(data):
    
    i = 0  

    subjects = []
    subject_idx = []
    readm_label = []
    mortality_label = []
    los_label = []
    cluster = []


    for group_idx, group_rows in data:  
        
        subjects.append(group_idx)
        subject_idx.append(i)
        
        readm_label.append(group_rows['readmission'].values[0])
        mortality_label.append(group_rows['mortality'].values[0])
        los_label.append(group_rows['length_of_stay'].values[0])
        cluster.append(group_rows['cluster'].values[0])
        
        
        
        # stores totals for variables
        cur_matrix = np.empty([48, 48])
        cur_matrix[:] = np.nan

        # stores counts for variables
        cur_counts = np.empty([48, 48])
        cur_counts[:] = np.nan

        cur_columns = group_rows.columns.values.tolist()
        feature_columns = cur_columns[3:-6]

        j = 0
        for idx, row in group_rows.iterrows():
            
                
            # Modifying cur_data to have data by the hour for 48 hours
            if row['Hours'] < j+1 and j < 48:
                for k in range(len(feature_columns)):
                    if not (np.isnan(group_rows.loc[idx, feature_columns[k]])):
                        if np.isnan(cur_matrix[j, k]):
                            cur_matrix[j, k] = group_rows.loc[idx, feature_columns[k]]
                            cur_counts[j, k] = 1
                        else:
                            cur_matrix[j, k] += group_rows.loc[idx, feature_columns[k]]
                            cur_counts[j, k] += 1
                            
            else:
                if j >= 48:
                    break
                else:
                    j += 1
                    

        # Getting time series data

        X_element = np.divide(cur_matrix, cur_counts)

        if i == 0:

            # Holds all of the multivariate time series
            X = np.array([X_element])

        else:
            X = np.concatenate((X, np.array([X_element])))
        
        i += 1



    y = pd.DataFrame({'subject':subjects, 'subject_idx':subject_idx, 'readmission':readm_label, 'mortality':mortality_label,
                  'length_of_stay':los_label, 'cluster':cluster})

    print(y.shape)
    y.head()    

    return X, y


def extract_obs_seq(X):
    # Extracting more observed sequences



    idxs = []

    props = []

    props_in_sample = []

    for i in range(X.shape[0]):
        
        
        cur_matrix = X[i,:,:]

        flattened_matrix = cur_matrix.flatten()
        
        flattened_series = pd.Series(flattened_matrix)

        prop_unobserved = len(flattened_series.loc[np.isnan(flattened_series)]) / len(flattened_series) 
        props.append(prop_unobserved)

        if prop_unobserved < 0.2:
            idxs.append(i)
            props_in_sample.append(prop_unobserved)

    X = X[idxs,:,:]
            
    print(len(idxs))
    print("Overall missingness: ", np.mean(props), "\n")
    print("Missingness in more observed samples: ", np.mean(props_in_sample), "\n")

    return X


def create_std_data(X, X_train, X_test):

    scalers = {}
    for i in range(X_train.shape[2]):
        scalers[i] = sklearn.preprocessing.StandardScaler()
        X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i]) 

    for i in range(X_test.shape[2]):
        X_test[:, :, i] = scalers[i].transform(X_test[:, :, i]) 

    for i in range(X.shape[2]):
        X[:, :, i] = scalers[i].transform(X[:, :, i]) 

    return X, X_train, X_test, scalers


def create_minmax_data(X_train_minmax, X_test_minmax):

    train_minmax_scalers = {}



    for i in range(X_train_minmax.shape[2]):
        train_minmax_scalers[i] = sklearn.preprocessing.MinMaxScaler()
        X_train_minmax[:, :, i] = train_minmax_scalers[i].fit_transform(X_train_minmax[:, :, i]) 

    for i in range(X_test_minmax.shape[2]):
        X_test_minmax[:, :, i] = train_minmax_scalers[i].transform(X_test_minmax[:, :, i]) 

   
    return X_train_minmax, X_test_minmax, train_minmax_scalers

def get_feature_means(X):

    all_feature_means = []

    # Reshaping to 2-dimensional data for imputation
    X_2d = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))


    mean_imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    mean_imputer.fit(X_2d)
    X_2d = mean_imputer.transform(X_2d)


    for i in range(X_2d.shape[1]):
        all_feature_means.append(np.mean(X_2d[:][i]))

    return all_feature_means
    


def create_mean_imputed_data(X_train, X_test, train_feature_means, test_feature_means):
    impute_value = 0.

    X_train_imputed = np.empty([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
    X_test_imputed = np.empty([X_test.shape[0], X_test.shape[1], X_test.shape[2]])

    train_mask = np.empty([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
    test_mask = np.empty([X_test.shape[0], X_test.shape[1], X_test.shape[2]])


    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            for k in range(X_train.shape[2]):
                if np.isnan(X_train[i,j,k]):
                    X_train_imputed[i,j,k] = train_feature_means[k]
                    train_mask[i,j,k] = 0
                else:
                    X_train_imputed[i,j,k] = X_train[i,j,k]
                    train_mask[i,j,k] = 1


    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            for k in range(X_test.shape[2]):
                if np.isnan(X_test[i,j,k]):
                    X_test_imputed[i,j,k] = train_feature_means[k]
                    test_mask[i,j,k] = 0
                else:
                    X_test_imputed[i,j,k] = X_test[i,j,k]
                    test_mask[i,j,k] = 1

    return X_train_imputed, X_test_imputed
    

def get_miss_forest_imputer(X_train_minmax):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer  
    from sklearn.impute import IterativeImputer


    n_est = 8
    max_depth = 25

    estimator =  RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, bootstrap=True, 
                                    random_state=random_seed)
    X_train_2d = np.reshape(X_train_minmax, (X_train_minmax.shape[0]*X_train_minmax.shape[1], X_train_minmax.shape[2]))

    miss_forest_imputer = sklearn.impute.IterativeImputer(random_state=random_seed, estimator=estimator, max_iter=10)

    miss_forest_imputer.fit(X_train_2d)

    return miss_forest_imputer

def create_mf_imputed_data(X_train, X_test, miss_forest_imputer):
    

    X_train_2d = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], X_train.shape[2]))

    X_train_imputed_2d = miss_forest_imputer.transform(X_train_2d)
    X_train_imputed = np.reshape(X_train_imputed_2d, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    

    X_test_2d = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2])) 

    X_test_imputed_2d = miss_forest_imputer.transform(X_test_2d)
    X_test_imputed = np.reshape(X_test_imputed_2d, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    return X_train_imputed, X_test_imputed


def vae_preprocessing(X_train, X_test):
    impute_value = 0.

    X_train_imputed = np.empty([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
    X_test_imputed = np.empty([X_test.shape[0], X_test.shape[1], X_test.shape[2]])

    train_mask = np.empty([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
    test_mask = np.empty([X_test.shape[0], X_test.shape[1], X_test.shape[2]])


    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            for k in range(X_train.shape[2]):
                if np.isnan(X_train[i,j,k]):
                    X_train_imputed[i,j,k] = impute_value
                    train_mask[i,j,k] = 0
                else:
                    X_train_imputed[i,j,k] = X_train[i,j,k]
                    train_mask[i,j,k] = 1


    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            for k in range(X_test.shape[2]):
                if np.isnan(X_test[i,j,k]):
                    X_test_imputed[i,j,k] = impute_value
                    test_mask[i,j,k] = 0
                else:
                    X_test_imputed[i,j,k] = X_test[i,j,k]
                    test_mask[i,j,k] = 1
                    
    return X_train_imputed, X_test_imputed, train_mask, test_mask



def train_eval_vae_model(model, processed_X_train, processed_X_test, train_mask, test_mask, batch_size):
  
    es = EarlyStopping(patience=10, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
    model.fit([processed_X_train, train_mask], batch_size=batch_size, validation_split=0.2, epochs=100, shuffle=False, callbacks=[es])


    vae = tf.keras.Model(model.input, model.output)

    reconstruc_train = vae.predict([processed_X_train,  train_mask])
    reconstruc_test = vae.predict([processed_X_test, test_mask])

    mse = 0
    mae = 0

    #print(mask_X_test_imputed[1])
    masked_reconstruction = tf.math.multiply(reconstruc_test, test_mask)



    for i in range(processed_X_test.shape[0]):
        mse += np.sum(np.square(np.subtract(processed_X_test[i], masked_reconstruction[i]))) / np.sum(test_mask[i])
        mae += np.sum(np.absolute(np.subtract(processed_X_test[i], masked_reconstruction[i]))) / np.sum(test_mask[i])


    print("test mse: ", mse / processed_X_test.shape[0])
    print("test mae: ", mae / processed_X_test.shape[0], "\n")

    return model, reconstruc_train, reconstruc_test


def imputed_vae_data(X_train, X_test, reconstruc_train, reconstruc_test):
  
    X_train_imputed = np.empty([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
    X_test_imputed = np.empty([X_test.shape[0], X_test.shape[1], X_test.shape[2]])


    # Impute original with reconstruction

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            for k in range(X_train.shape[2]):
                if np.isnan(X_train[i,j,k]):
                    X_train_imputed[i,j,k] = reconstruc_train[i,j,k]
                else:
                    X_train_imputed[i,j,k] = X_train[i,j,k]


    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            for k in range(X_test.shape[2]):
                if np.isnan(X_test[i,j,k]):
                    X_test_imputed[i,j,k] = reconstruc_test[i,j,k]
                else:
                    X_test_imputed[i,j,k] = X_test[i,j,k]

    return X_train_imputed, X_test_imputed


def all_eval(X, X_minmax, scalers, test_minmax_scalers, cnn_vae, lstm_vae, batch_size, miss_forest_imputer, train_feature_means, 
             knn_imputers, lstm_ae_model, mask_prop):
    # Creating a random mask for evaluation of imputation 

    iter = 30
    cnn_vae_mse_list = []
    cnn_vae_mae_list = []

    lstm_vae_mse_list = []
    lstm_vae_mae_list = []
    
    mf_mse_list = []
    mf_mae_list = []
    
    mean_mse_list = []
    mean_mae_list = []
    
    fill_mse_list = []
    fill_mae_list = []
    
    dynimp_mse_list = []
    dynimp_mae_list = []
    

    n_samples = X.shape[0]

    rand_X_imputed = np.empty([X.shape[0], X.shape[1], X.shape[2]])
    mask_X_imputed = np.empty([X.shape[0], X.shape[1], X.shape[2]])
    rand_mask = np.empty([X.shape[0], X.shape[1], X.shape[2]])

    for j in range(iter):

        for i in range(rand_X_imputed.shape[0]):
            rand_mask_1d = np.random.choice([0,1], size=X.shape[1]*X.shape[2], replace=True, p=[mask_prop, 1-mask_prop])
            rand_mask[i] = np.reshape(rand_mask_1d, (X.shape[1],X.shape[2]))

        rand_mask = np.where(np.isnan(X), 0, rand_mask)

        cnn_vae_mse = 0
        cnn_vae_mae = 0

        lstm_vae_mse = 0
        lstm_vae_mae = 0
        
        mf_mse = 0
        mf_mae = 0
        
        mean_mse = 0
        mean_mae = 0
        
        fill_mse = 0
        fill_mae = 0
        
        dynimp_mse = 0
        dynimp_mae = 0

        # Actual mask of observations for comparison
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)



        ############################################ 
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, 0, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        
        # Performing and evaluating cnn_vae imputation
        cnn_vae_imputated_data = cnn_vae.predict([rand_X_imputed, rand_mask], batch_size=batch_size)


        # Only considering observations where actual was randomly masked out
        cnn_vae_imputated_data = np.where(rand_mask==0, cnn_vae_imputated_data, 0)
        cnn_vae_imputated_data = np.where(np.isnan(X), 0, cnn_vae_imputated_data)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)

        for i in range(rand_X_imputed.shape[0]):
            cnn_vae_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], cnn_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])
            cnn_vae_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], cnn_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])

        cnn_vae_mse_list.append(cnn_vae_mse / n_samples)
        cnn_vae_mae_list.append(cnn_vae_mae / n_samples)

        ############################################ 
        
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)

        # lstm_vae imputation
        rand_X_imputed = np.where(rand_mask==0, 0, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        
        # Performing and evaluating lstm_vae imputation
        lstm_vae_imputated_data = lstm_vae.predict([rand_X_imputed, rand_mask], batch_size=batch_size)


        # Only considering observations where actual was randomly masked out
        lstm_vae_imputated_data = np.where(rand_mask==0, lstm_vae_imputated_data, 0)
        lstm_vae_imputated_data = np.where(np.isnan(X), 0, lstm_vae_imputated_data)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)

        for i in range(rand_X_imputed.shape[0]):
            lstm_vae_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], lstm_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])
            lstm_vae_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], lstm_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])

        lstm_vae_mse_list.append(lstm_vae_mse / n_samples)
        lstm_vae_mae_list.append(lstm_vae_mae / n_samples)
        
        ############################################ 
        
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        
        # Performing and evaluating missforest imputation
        
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        # imputing on the random mask data
        rand_X_imputed_2d = np.reshape(rand_X_imputed, (rand_X_imputed.shape[0]*rand_X_imputed.shape[1], rand_X_imputed.shape[2]))

        mf_X_imputed_2d = miss_forest_imputer.transform(rand_X_imputed_2d)
        mf_X_imputed = np.reshape(mf_X_imputed_2d, (rand_X_imputed.shape[0], rand_X_imputed.shape[1], rand_X_imputed.shape[2]))

        #print(rand_X_test_imputed)

        # Only considering observations where actual was randomly masked out
        mf_X_imputed = np.where(rand_mask==0, mf_X_imputed, 0)
        mf_X_imputed = np.where(np.isnan(X), 0, mf_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            mf_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], mf_X_imputed[i]))) / np.sum(rand_error_mask[i])
            mf_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], mf_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        mf_mse_list.append(mf_mse / n_samples)
        mf_mae_list.append(mf_mae / n_samples)




        ############################################ 
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        
        # Performing and evaluating mean imputation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        mean_X_imputed = np.where(np.isnan(rand_X_imputed), train_feature_means,  rand_X_imputed)
        
        
        mean_X_imputed = np.where(rand_mask==0, mean_X_imputed, 0)
        mean_X_imputed = np.where(np.isnan(X), 0, mean_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            mean_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], mean_X_imputed[i]))) / np.sum(rand_error_mask[i])
            mean_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], mean_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        mean_mse_list.append(mean_mse / n_samples)
        mean_mae_list.append(mean_mae / n_samples)

        ############################################ 
         # Actual mask of observations for comparison
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        # imputing on the random mask data
        
        for k in range(rand_X_imputed.shape[0]):
            intermediate_df = pd.DataFrame(rand_X_imputed[k,:,:])
            intermediate_df = intermediate_df.ffill()
            intermediate_df = intermediate_df.bfill()
            intermediate_array = intermediate_df.to_numpy()
            rand_X_imputed[k,:,:] = intermediate_array
        
        forward_X_imputed = np.where(np.isnan(rand_X_imputed), train_feature_means,  rand_X_imputed)

        #print(rand_X_test_imputed)

        # Only considering observations where actual was randomly masked out
        forward_X_imputed = np.where(rand_mask==0, forward_X_imputed, 0)
        forward_X_imputed = np.where(np.isnan(X), 0, forward_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            fill_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], forward_X_imputed[i]))) / np.sum(rand_error_mask[i])
            fill_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], forward_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        fill_mse_list.append(fill_mse / n_samples)
        fill_mae_list.append(fill_mae / n_samples)

        ############################################ 
         # Actual mask of observations for comparison
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        # imputing on the random mask data
        
        rand_X_imputed_knn = np.empty([rand_X_imputed.shape[0], rand_X_imputed.shape[1], rand_X_imputed.shape[2]])
        rand_X_imputed_transposed = rand_X_imputed.transpose((1, 0, 2))

        for i in range(rand_X_imputed_transposed.shape[0]):

            cur_imputer = knn_imputers[i]

            rand_X_imputed_transposed[i] = cur_imputer.transform(rand_X_imputed_transposed[i])

        rand_X_imputed_knn = rand_X_imputed_transposed.transpose((1, 0, 2))

        dynimp_X_imputed = lstm_ae_model.predict(rand_X_imputed_knn, batch_size=batch_size)
        

        #print(rand_X_test_imputed)
        

        # Only considering observations where actual was randomly masked out
        dynimp_X_imputed = np.where(rand_mask==0, dynimp_X_imputed, 0)
        dynimp_X_imputed = np.where(np.isnan(X_minmax), 0, dynimp_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X_minmax), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X_minmax), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            dynimp_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], dynimp_X_imputed[i]))) / np.sum(rand_error_mask[i])
            dynimp_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], dynimp_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        dynimp_mse_list.append(dynimp_mse / n_samples)
        dynimp_mae_list.append(dynimp_mae / n_samples)

    print("fill mse:")
    print("mean: ", np.mean(fill_mse_list))
    print("std dev: ",np.std(fill_mse_list), "\n")


    print("fill_mae:")
    print("mean: ", np.mean(fill_mae_list))
    print("std dev: ",np.std(fill_mae_list), "\n")

        

        


    print("cnn-vae imputation mse:")
    print("mean: ", np.mean(cnn_vae_mse_list))
    print("std dev: ", np.std(cnn_vae_mse_list), "\n")

    print("cnn-vae imputation mae:")
    print("mean: ", np.mean(cnn_vae_mae_list))
    print("std dev: ", np.std(cnn_vae_mae_list), "\n\n")



    print("lstm-vae imputation mse:")
    print("mean: ", np.mean(lstm_vae_mse_list))
    print("std dev: ", np.std(lstm_vae_mse_list), "\n")

    print("lstm-vae imputation mae:")
    print("mean: ", np.mean(lstm_vae_mae_list))
    print("std dev: ", np.std(lstm_vae_mae_list), "\n\n")


    print("missforest imputation mse:")
    print("mean: ", np.mean(mf_mse_list))
    print("std dev: ", np.std(mf_mse_list), "\n")
    
    print("missforest imputation mae:")
    print("mean: ", np.mean(mf_mae_list))
    print("std dev: ", np.std(mf_mae_list), "\n\n")


    
    print("mean imputation mse:")
    print("mean: ", np.mean(mean_mse_list))
    print("std dev: ", np.std(mean_mse_list), "\n")
    
    print("mean imputation mae:")
    print("mean: ", np.mean(mean_mae_list))
    print("std dev: ", np.std(mean_mae_list), "\n\n")
    
    
    print("dynimp imputation mse:")
    print("mean: ", np.mean(dynimp_mse_list))
    print("std dev: ", np.std(dynimp_mse_list), "\n")
    
    print("dynimp imputation mae:")
    print("mean: ", np.mean(dynimp_mae_list))
    print("std dev: ", np.std(dynimp_mae_list), "\n\n")


# dynimp

# Create train set with noise

def create_X_train_w_noise(X_train, mask_prop):
    
    X_train_noise = np.empty([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
    rand_mask = np.empty([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
    
    for i in range(X_train.shape[0]):
        rand_mask_1d = np.random.choice([0,1], size=X_train.shape[1]*X_train.shape[2], replace=True, p=[mask_prop, 1-mask_prop])
        rand_mask[i] = np.reshape(rand_mask_1d, (X_train.shape[1], X_train.shape[2]))

    rand_mask = np.where(np.isnan(X_train), 0, rand_mask)
             
    X_train_noise = np.where(rand_mask==0, np.nan, X_train)
    
    return X_train_noise
    
def knn_impute_data(X_train_w_noise):
    # Perform KNN on train set with noise

    knn_imputers = {}

    X_train_w_noise_transposed = X_train_w_noise.transpose((1, 0, 2))

    for i in range(X_train_w_noise_transposed.shape[0]):
        
        cur_imputer = sklearn.impute.KNNImputer(n_neighbors=5)
        
        X_train_w_noise_transposed[i] = cur_imputer.fit_transform(X_train_w_noise_transposed[i])
        
        knn_imputers[i] = cur_imputer
        

    X_train_noise = X_train_w_noise_transposed.transpose((1, 0, 2))

    return X_train_noise, knn_imputers

def knn_impute_data_w_trained(X_test_minmax, knn_imputers):

    X_test_knn = knn_impute_data(X_test_minmax)

    X_test_knn = np.empty([X_test_minmax.shape[0], X_test_minmax.shape[1], X_test_minmax.shape[2]])

    X_test_transposed = X_test_minmax.transpose((1, 0, 2))

    for i in range(X_test_transposed.shape[0]):
        
        cur_imputer = knn_imputers[i]
        
        X_test_transposed[i] = cur_imputer.transform(X_test_transposed[i])
        
            
    X_test_knn = X_test_transposed.transpose((1, 0, 2))

    return X_test_knn


def all_eval_v2(X, X_minmax, scalers, test_minmax_scalers, cnn_vae, lstm_vae, batch_size, miss_forest_imputer, train_feature_means, 
             knn_imputers, lstm_ae_model, mask_prop):
    # Creating a random mask for evaluation of imputation 

    iter = 30
    cnn_vae_mse_list = []
    cnn_vae_mae_list = []

    lstm_vae_mse_list = []
    lstm_vae_mae_list = []
    
    mf_mse_list = []
    mf_mae_list = []
    
    mean_mse_list = []
    mean_mae_list = []
    
    fill_mse_list = []
    fill_mae_list = []
    
    dynimp_mse_list = []
    dynimp_mae_list = []
    

    n_samples = X.shape[0]

    rand_X_imputed = np.empty([X.shape[0], X.shape[1], X.shape[2]])
    mask_X_imputed = np.empty([X.shape[0], X.shape[1], X.shape[2]])
    rand_mask = np.empty([X.shape[0], X.shape[1], X.shape[2]])

    rand_mask_time = np.empty([X.shape[0], X.shape[1], X.shape[2]])
    rand_mask_features = np.empty([X.shape[0], X.shape[1], X.shape[2]])

    for j in range(iter):

        for i in range(rand_X_imputed.shape[0]):
            # Inducing randomness across channels
            for k in range(rand_X_imputed.shape[1]):
                rand_mask_1d = np.random.choice([0,1], X.shape[2], replace=True, p=[mask_prop, 1-mask_prop])
                rand_mask_time[i,k] = rand_mask_1d
                
            # Inducing randomness across features            
            for l in range(rand_X_imputed.shape[2]):   
                rand_mask_1d = np.random.choice([0,1], X.shape[1], replace=True, p=[mask_prop, 1-mask_prop])
                rand_mask_features[i,:,l] = rand_mask_1d
                
        rand_mask = np.logical_or(rand_mask_time, rand_mask_features)  
        
        rand_mask = np.where(np.isnan(X), 0, rand_mask)


        cnn_vae_mse = 0
        cnn_vae_mae = 0

        lstm_vae_mse = 0
        lstm_vae_mae = 0
        
        mf_mse = 0
        mf_mae = 0
        
        mean_mse = 0
        mean_mae = 0
        
        fill_mse = 0
        fill_mae = 0
        
        dynimp_mse = 0
        dynimp_mae = 0

        # Actual mask of observations for comparison
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)



        ############################################ 
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, 0, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        
        # Performing and evaluating cnn_vae imputation
        cnn_vae_imputated_data = cnn_vae.predict([rand_X_imputed, rand_mask], batch_size=batch_size)


        # Only considering observations where actual was randomly masked out
        cnn_vae_imputated_data = np.where(rand_mask==0, cnn_vae_imputated_data, 0)
        cnn_vae_imputated_data = np.where(np.isnan(X), 0, cnn_vae_imputated_data)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)

        for i in range(rand_X_imputed.shape[0]):
            cnn_vae_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], cnn_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])
            cnn_vae_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], cnn_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])

        cnn_vae_mse_list.append(cnn_vae_mse / n_samples)
        cnn_vae_mae_list.append(cnn_vae_mae / n_samples)

        ############################################ 
        
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)

        # lstm_vae imputation
        rand_X_imputed = np.where(rand_mask==0, 0, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        
        # Performing and evaluating lstm_vae imputation
        lstm_vae_imputated_data = lstm_vae.predict([rand_X_imputed, rand_mask], batch_size=batch_size)


        # Only considering observations where actual was randomly masked out
        lstm_vae_imputated_data = np.where(rand_mask==0, lstm_vae_imputated_data, 0)
        lstm_vae_imputated_data = np.where(np.isnan(X), 0, lstm_vae_imputated_data)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)

        for i in range(rand_X_imputed.shape[0]):
            lstm_vae_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], lstm_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])
            lstm_vae_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], lstm_vae_imputated_data[i]))) / np.sum(rand_error_mask[i])

        lstm_vae_mse_list.append(lstm_vae_mse / n_samples)
        lstm_vae_mae_list.append(lstm_vae_mae / n_samples)
        
        ############################################ 
        
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        
        # Performing and evaluating missforest imputation
        
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        # imputing on the random mask data
        rand_X_imputed_2d = np.reshape(rand_X_imputed, (rand_X_imputed.shape[0]*rand_X_imputed.shape[1], rand_X_imputed.shape[2]))

        mf_X_imputed_2d = miss_forest_imputer.transform(rand_X_imputed_2d)
        mf_X_imputed = np.reshape(mf_X_imputed_2d, (rand_X_imputed.shape[0], rand_X_imputed.shape[1], rand_X_imputed.shape[2]))

        #print(rand_X_test_imputed)

        # Only considering observations where actual was randomly masked out
        mf_X_imputed = np.where(rand_mask==0, mf_X_imputed, 0)
        mf_X_imputed = np.where(np.isnan(X), 0, mf_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            mf_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], mf_X_imputed[i]))) / np.sum(rand_error_mask[i])
            mf_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], mf_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        mf_mse_list.append(mf_mse / n_samples)
        mf_mae_list.append(mf_mae / n_samples)




        ############################################ 
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        
        # Performing and evaluating mean imputation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        mean_X_imputed = np.where(np.isnan(rand_X_imputed), train_feature_means,  rand_X_imputed)
        
        
        mean_X_imputed = np.where(rand_mask==0, mean_X_imputed, 0)
        mean_X_imputed = np.where(np.isnan(X), 0, mean_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            mean_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], mean_X_imputed[i]))) / np.sum(rand_error_mask[i])
            mean_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], mean_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        mean_mse_list.append(mean_mse / n_samples)
        mean_mae_list.append(mean_mae / n_samples)

        ############################################ 
         # Actual mask of observations for comparison
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        # imputing on the random mask data
        
        for k in range(rand_X_imputed.shape[0]):
            intermediate_df = pd.DataFrame(rand_X_imputed[k,:,:])
            intermediate_df = intermediate_df.ffill()
            intermediate_df = intermediate_df.bfill()
            intermediate_array = intermediate_df.to_numpy()
            rand_X_imputed[k,:,:] = intermediate_array
        
        forward_X_imputed = np.where(np.isnan(rand_X_imputed), train_feature_means,  rand_X_imputed)

        #print(rand_X_test_imputed)

        # Only considering observations where actual was randomly masked out
        forward_X_imputed = np.where(rand_mask==0, forward_X_imputed, 0)
        forward_X_imputed = np.where(np.isnan(X), 0, forward_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            fill_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], forward_X_imputed[i]))) / np.sum(rand_error_mask[i])
            fill_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], forward_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        fill_mse_list.append(fill_mse / n_samples)
        fill_mae_list.append(fill_mae / n_samples)

        ############################################ 
         # Actual mask of observations for comparison
        mask_X_imputed = np.where(np.isnan(X), 0, X)
        mask_X = np.where(np.isnan(X), 0, 1)
        # Random mask for evaluation
        rand_X_imputed = np.where(rand_mask==0, np.nan, mask_X_imputed)
        #print(rand_X_test_imputed[i])

        #print(mask_X_test_imputed[i].shape)

        # imputing on the random mask data
        
        rand_X_imputed_knn = np.empty([rand_X_imputed.shape[0], rand_X_imputed.shape[1], rand_X_imputed.shape[2]])
        rand_X_imputed_transposed = rand_X_imputed.transpose((1, 0, 2))

        for i in range(rand_X_imputed_transposed.shape[0]):

            cur_imputer = knn_imputers[i]

            rand_X_imputed_transposed[i] = cur_imputer.transform(rand_X_imputed_transposed[i])

        rand_X_imputed_knn = rand_X_imputed_transposed.transpose((1, 0, 2))

        dynimp_X_imputed = lstm_ae_model.predict(rand_X_imputed_knn, batch_size=batch_size)

        #print(rand_X_test_imputed)
        

        # Only considering observations where actual was randomly masked out
        dynimp_X_imputed = np.where(rand_mask==0, dynimp_X_imputed, 0)
        dynimp_X_imputed = np.where(np.isnan(X_minmax), 0, dynimp_X_imputed)

        mask_X_imputed = np.where(rand_mask==0, mask_X_imputed, 0)
        mask_X_imputed = np.where(np.isnan(X_minmax), 0, mask_X_imputed)

        rand_error_mask = np.where(rand_mask==0, 1, 0)
        rand_error_mask = np.where(np.isnan(X_minmax), 0, 1)



        for i in range(rand_X_imputed.shape[0]):
            dynimp_mse += np.sum(np.square(np.subtract(mask_X_imputed[i], dynimp_X_imputed[i]))) / np.sum(rand_error_mask[i])
            dynimp_mae += np.sum(np.absolute(np.subtract(mask_X_imputed[i], dynimp_X_imputed[i]))) / np.sum(rand_error_mask[i])

            
        dynimp_mse_list.append(dynimp_mse / n_samples)
        dynimp_mae_list.append(dynimp_mae / n_samples)

    print("fill mse:")
    print("mean: ", np.mean(fill_mse_list))
    print("std dev: ",np.std(fill_mse_list), "\n")


    print("fill_mae:")
    print("mean: ", np.mean(fill_mae_list))
    print("std dev: ",np.std(fill_mae_list), "\n")

        

        


    print("cnn-vae imputation mse:")
    print("mean: ", np.mean(cnn_vae_mse_list))
    print("std dev: ", np.std(cnn_vae_mse_list), "\n")

    print("cnn-vae imputation mae:")
    print("mean: ", np.mean(cnn_vae_mae_list))
    print("std dev: ", np.std(cnn_vae_mae_list), "\n\n")



    print("lstm-vae imputation mse:")
    print("mean: ", np.mean(lstm_vae_mse_list))
    print("std dev: ", np.std(lstm_vae_mse_list), "\n")

    print("lstm-vae imputation mae:")
    print("mean: ", np.mean(lstm_vae_mae_list))
    print("std dev: ", np.std(lstm_vae_mae_list), "\n\n")


    print("missforest imputation mse:")
    print("mean: ", np.mean(mf_mse_list))
    print("std dev: ", np.std(mf_mse_list), "\n")
    
    print("missforest imputation mae:")
    print("mean: ", np.mean(mf_mae_list))
    print("std dev: ", np.std(mf_mae_list), "\n\n")


    
    print("mean imputation mse:")
    print("mean: ", np.mean(mean_mse_list))
    print("std dev: ", np.std(mean_mse_list), "\n")
    
    print("mean imputation mae:")
    print("mean: ", np.mean(mean_mae_list))
    print("std dev: ", np.std(mean_mae_list), "\n\n")
    
    
    print("dynimp imputation mse:")
    print("mean: ", np.mean(dynimp_mse_list))
    print("std dev: ", np.std(dynimp_mse_list), "\n")
    
    print("dynimp imputation mae:")
    print("mean: ", np.mean(dynimp_mae_list))
    print("std dev: ", np.std(dynimp_mae_list), "\n\n")


def readm_preprocessing(X_train_imputed, X_test_imputed, y_train, y_test):
    readm_X_train = np.empty([X_train_imputed.shape[0], X_train_imputed.shape[1], X_train_imputed.shape[2]])
    readm_X_test = np.empty([X_test_imputed.shape[0], X_test_imputed.shape[1], X_test_imputed.shape[2]])

    for i in range(X_train_imputed.shape[0]):
        for j in range(X_train_imputed.shape[1]):
            for k in range(X_train_imputed.shape[2]):
                readm_X_train[i,j,k] = X_train_imputed[i,j,k]



    for i in range(X_test_imputed.shape[0]):
        for j in range(X_test_imputed.shape[1]):
            for k in range(X_test_imputed.shape[2]):
                readm_X_test[i,j,k] = X_test_imputed[i,j,k]



    readm_y_train = y_train['readmission']
    readm_y_test = y_test['readmission']


    rm_idx_train = []
    rm_idx_test = []


    for i in range(readm_X_train.shape[0]):
        if np.isnan(y_train['readmission'].values[i]) or y_train['mortality'].values[i] == 1:
            rm_idx_train.append(i)

    for i in range(readm_X_test.shape[0]):
        if np.isnan(y_test['readmission'].values[i]) or y_train['mortality'].values[i] == 1:
            rm_idx_test.append(i)

    readm_X_train = np.delete(readm_X_train, rm_idx_train, 0)
    readm_y_train = np.delete(np.array(readm_y_train), rm_idx_train, 0)

    readm_X_test = np.delete(readm_X_test, rm_idx_test, 0)
    readm_y_test = np.delete(np.array(readm_y_test), rm_idx_test, 0)

    #print(readm_X_train.shape)
    #print(readm_y_train.shape)

    #print(readm_X_test.shape)
    #print(readm_y_test.shape)

    #print(np.where(readm_y_train == 1))
    #print(np.where(readm_y_test == 1))  
    return readm_X_train, readm_X_test, readm_y_train, readm_y_test


def mortality_preprocessing(X_train_imputed, X_test_imputed, y_train, y_test):
    # Processing Data for Mortality
    mortality_X_train = np.empty([X_train_imputed.shape[0], X_train_imputed.shape[1], X_train_imputed.shape[2]])
    mortality_X_test = np.empty([X_test_imputed.shape[0], X_test_imputed.shape[1], X_test_imputed.shape[2]])

    for i in range(X_train_imputed.shape[0]):
        for j in range(X_train_imputed.shape[1]):
            for k in range(X_train_imputed.shape[2]):
                mortality_X_train[i,j,k] = X_train_imputed[i,j,k]


    for i in range(X_test_imputed.shape[0]):
        for j in range(X_test_imputed.shape[1]):
            for k in range(X_test_imputed.shape[2]):
                mortality_X_test[i,j,k] = X_test_imputed[i,j,k]

    mortality_y_train = y_train['mortality']
    mortality_y_test = y_test['mortality']


    #print(np.where(mortality_y_train == 1))
    #print(np.where(mortality_y_test == 1))
    return mortality_X_train, mortality_X_test, mortality_y_train, mortality_y_test


def los_preprocessing(X_train_imputed, X_test_imputed, y_train, y_test):
    # Processing Data for Length of Stay
    los_X_train = np.empty([X_train_imputed.shape[0], X_train_imputed.shape[1], X_train_imputed.shape[2]])
    los_X_test = np.empty([X_test_imputed.shape[0], X_test_imputed.shape[1], X_test_imputed.shape[2]])

    for i in range(X_train_imputed.shape[0]):
        for j in range(X_train_imputed.shape[1]):
            for k in range(X_train_imputed.shape[2]):
                los_X_train[i,j,k] = X_train_imputed[i,j,k]


    for i in range(X_test_imputed.shape[0]):
        for j in range(X_test_imputed.shape[1]):
            for k in range(X_test_imputed.shape[2]):
                los_X_test[i,j,k] = X_test_imputed[i,j,k]

                
    los_y_train = y_train['length_of_stay']
    los_y_test = y_test['length_of_stay']
    
    rm_idx_train = []
    rm_idx_test = []


    for i in range(los_X_train.shape[0]):
        if los_y_train.values[i] < 0:
            rm_idx_train.append(i)

    for i in range(los_X_test.shape[0]):
        if los_y_test.values[i] < 0:
            rm_idx_test.append(i)

    los_X_train = np.delete(los_X_train, rm_idx_train, 0)
    los_y_train = np.delete(np.array(los_y_train), rm_idx_train, 0)

    los_X_test = np.delete(los_X_test, rm_idx_test, 0)
    los_y_test = np.delete(np.array(los_y_test), rm_idx_test, 0)
    
          
    los_y_train = (los_y_train - np.full(len(los_y_train), np.mean(los_y_train))) / np.std(los_y_train)
    
    los_y_test = (los_y_test - np.full(len(los_y_test), np.mean(los_y_test))) / np.std(los_y_test)
  

    return los_X_train, los_X_test, los_y_train, los_y_test


def create_class_model():
  es = EarlyStopping(patience=20, verbose=1, min_delta=0.0001, monitor='val_auc', mode='auto', restore_best_weights=True)

  class_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
  ])

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

  class_model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'), 
                        tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

  return class_model


def create_reg_model():
    # LSTM Regression Model

    es = EarlyStopping(patience=10, verbose=1, min_delta=0.0001, monitor='val_loss', mode='auto', restore_best_weights=True)

    reg_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='relu'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    reg_model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return reg_model


def train_eval_pred_model(model, batch_size, epochs, X_train, X_test, y_train, y_test):
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2, callbacks=[es])
    
    predictions = model.predict(X_test, batch_size=batch_size)

    metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
    
    return model, predictions



# Implementing Logistic Regression Model
def train_test_lr_model(X_train, X_test, y_train, y_test): 
    
    lr_model = sklearn.linear_modelLogisticRegression(random_state=random_seed, max_iter=10e6)
    
    lr_model.fit(X_train, y_train)
    
    preds = lr_model.predict_proba(X_test)
    
    auroc = sklearn.metrics.roc_auc_score(y_test, preds[:,1])
    print("auroc: ", auroc, "\n")
    
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, preds[:,1])
    
    auprc = sklearn.metrics.auc(recall, precision)
    print("auprc: ", auprc, "\n")


# Implementing XGBoost model
def train_test_XGBoost_class(X_train, X_test, y_train, y_test):
    xg_class = xgb.XGBClassifier(objective ='binary:logistic', nthread=1, learning_rate = 0.2,
                    max_depth = 12, alpha = 12, n_estimators = 35, use_label_encoder = False, eval_metric='logloss')
    
    xg_class.fit(X_train,y_train)

    preds = xg_class.predict_proba(X_test)
    
    auroc = sklearn.metrics.roc_auc_score(y_test, preds[:,1])
    print("auroc: ", auroc, "\n")
    
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, preds[:,1])
    
    auprc = sklearn.metrics.auc(recall, precision)
    print("auprc: ", auprc, "\n")
    

def train_test_XGBoost_reg(X_train, X_test, y_train, y_test):
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', nthread=1, learning_rate = 0.2,
                    max_depth = 12, alpha = 12, n_estimators = 30)
    
    xg_reg.fit(X_train,y_train)

    preds = xg_reg.predict(X_test)
    
    mse = sklearn.metrics.mean_squared_error(y_test, preds)
    print("mse: ", mse, "\n")
    
    mae = sklearn.metrics.mean_absolute_error(y_test, preds)
    print("mae: ", mae, "\n")
    
   