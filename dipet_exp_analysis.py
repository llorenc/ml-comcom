# coding: utf-8
# Copyright (c) 2022  Gabriel Iuhasz and Llorenç Cerdà-Alabern
# script used to process the dataset in the paper "Anomaly
# Detection for Fault Detection in Wireless Community Networks Using
# Machine Learning"

# In[2]:
import os
import numpy as np
np.random.seed(42)
import importlib
import seaborn as sns
from sklearn.decomposition import SparsePCA, PCA
from sklearn.base import clone
from collections import Counter
import pandas as pd
import tqdm
import glob
import fnmatch
import re
import sys
import random
import itertools
# Import all models
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.lscp import LSCP
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
import shap
import time
import pickle

from subprocess import check_output
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()  # interactive non-blocking mode
matplotlib.rcParams.update({'font.size': 20})

# wd
pwd = os.getcwd()
print('pwd: ' + pwd)
# local modules
imported = {}
def force_import(name):
    global imported
    if name not in imported:
        imported[name] = importlib.import_module(name)
    else:
        importlib.reload(imported[name])
    return imported[name]

dl = force_import("dipet_exp_analysis_lib")

# Load data

anom = ['2021-04-14 01:55:00', '2021-04-14 18:10:00']
df_paper = dl.load_hd_data('training_jcsd2021.gz') 
df_paper_test = dl.load_hd_data('testing_jcsd2021.gz') 
df_paper.shape
df_paper_test.shape

# In[4]:
df_paper, rm_col = dl.filter_low_variance(df_paper)
df_paper_test = df_paper_test.drop(rm_col, axis=1)

dl.filter_nan_col(df_paper, drop_threashold=1, verbose=False)
dl.filter_nan_col(df_paper_test, drop_threashold=1, verbose=False)

#
# pca
#
df_paper, rm_col = dl.filter_low_variance(df_paper)
df_paper_test = df_paper_test.drop(rm_col, axis=1)

dl.filter_nan_col(df_paper, drop_threashold=1, verbose=False)
dl.filter_nan_col(df_paper_test, drop_threashold=1, verbose=False)

#
# pca, after filter_low_variance, using meanmax
#
df_paper_pca = df_paper.copy()
df_paper_test_pca = df_paper_test.copy()

df_paper_pca, rm_col = dl.filter_low_variance(df_paper_pca)
df_paper_test_pca = df_paper_test.drop(rm_col, axis=1)

scaler_pca = dl.MeshmonTransformer()
scaler_pca.fit(df_paper_pca, 'meanmax')
# scaler_pca.fit(df_paper_pca, 'minmax')
df_paper_pca = dl.transform_df(scaler_pca, df_paper_pca)
df_paper_test_pca = dl.transform_df(scaler_pca, df_paper_test_pca)

model = dl.PCAclf()
model.fit(df_paper_pca) # , var=0.80) #, pca_comp=45)
# sum(model.decision_function(df_paper_test_pca))
# model.QnQ

# Qscore = model.get_Q(df_paper_pca)

pkl = dl.build_model(clf_name='PCA', clf=model,
                     df=df_paper_pca,
                     df_testing=df_paper_test_pca)
dl.count_anom(pkl['y_pred_idx_test'], anom)

# note that points are distributted differently from the other ML
# methods due to meanmax scaling instead of minmax
dl.decision_boundary2(pkl, df_paper_pca, 
                      df_paper_test_pca, 
                      legend_loc='upper right', size=200
                      # ,file="Decision_Boundary_PCA.pdf"
                      )

# Scaling Data
scaler = dl.MeshmonTransformer()
scaler.fit(df_paper, 'minmax')

# Scaled training data back to df from np.array
df_filtered_scaled = dl.transform_df(scaler, df_paper)
df_filtered_scaled

# In[12]:
df_filtered_scaled_testing = dl.transform_df(scaler, df_paper_test)
df_filtered_scaled_testing

df_paper_test.isnull().values.any()
df_filtered_scaled_testing.isnull().values.any()
df_filtered_scaled_testing = df_filtered_scaled_testing.fillna(0)
df_filtered_scaled_testing.isnull().values.any()

# First pass using anomaly detection methods:
# Isolation Forest

# In[13]:
contamination = 0.005
max_features = 8
# contamination = "auto"

clf_if = IsolationForest(n_estimators=10,
                         warm_start=False,
                         contamination=contamination,
                         max_features=max_features,
                         random_state=np.random.RandomState(42))
clf_clone = clone(clf_if)
clf_if.fit(df_filtered_scaled)
pred = clf_if.predict(df_filtered_scaled)
print("Anomalies in dataset of size {} found: {}".
      format(len(list(pred)),list(pred).count(-1)))

# Get anomaly index and the values based on said index
anomaly_index_df = np.where(pred==-1)

transformer = PCA(n_components=2)
transformer.fit(df_filtered_scaled)
X_transformed = transformer.transform(df_filtered_scaled)
print("Initial shape: {}".format(df_filtered_scaled.shape))
print("PCA data shape: {}".format(X_transformed.shape))
# dataset_rf  = X_transformed

# Get anomalies based on index
values_rf = X_transformed[anomaly_index_df]

#Plot prediction on 2d
fig = plt.figure(figsize=(10, 8))
#fig = plt.figure(figsize=(15,15))
plt.scatter(X_transformed[:,0], X_transformed[:,1])
plt.scatter(values_rf[:,0], values_rf[:,1], color='r', marker='x')
plt.title("IsolationForest plot with a total of {} anomalies".
          format(list(pred).count(-1)))
plt.grid()
plt.show()

# In[14]:
# Data preprocessing options (3d)
transformer = PCA(n_components=3)
transformer.fit(df_filtered_scaled)
X_transformed_3d = transformer.transform(df_filtered_scaled)
print(X_transformed_3d.shape)

# Plot 3D anomaly map
#fig = plt.figure(figsize=(15,15))
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")
# Plot the compressed data points
ax.scatter(X_transformed_3d[:, 0], X_transformed_3d[:, 1], 
           zs=X_transformed_3d[:, 2], s=4, lw=1, label="normal",c="blue")
# Plot x's for the ground truth outliers
ax.scatter(X_transformed_3d[anomaly_index_df,0],
           X_transformed_3d[anomaly_index_df,1], X_transformed_3d[anomaly_index_df,2],
           lw=2, s=60, marker="x", c="red", label="anomalies")
ax.legend()
plt.show()

# In[15]:
df_anomalies_only = df_filtered_scaled.iloc[anomaly_index_df]
df_anomalies_only.shape

#%
# ff = filter_columns_wildcard(df_filtered_scaled,'sum', keep=True)
# plot_on_features(ff, pred)
# In[16]:
# Choose features to plot (Llorenc)
dl.l_plot(df=df_filtered_scaled, pred=pred, 
          title="IsolationForest plot with a total of {} anomalies".
          format(list(pred).count(-1)))

# Run on testing set

# In[17]:
# Window imp data range
# df_filtered_scaled_sample = df_filtered_scaled_testing.loc["2021-04-14 01:55:00":"2021-04-14 23:50:00"] # exact anomaly frames
df_filtered_scaled_sample = df_filtered_scaled_testing
pred_test = clf_if.predict(df_filtered_scaled_sample)
print("Anomalies in dataset of size {} found: {}".
      format(len(list(pred_test)),list(pred_test).count(-1)))

print(pred_test)
anomaly_index_df = np.where(pred_test==-1)
print(anomaly_index_df)
anomalies = df_filtered_scaled_sample.index[anomaly_index_df]

print(anomalies)
df_test = df_filtered_scaled_sample.copy(deep=True)
df_test['pred'] = pred_test

print(df_test[df_test['pred'] == -1])

# df_filtered_scaled_sample.loc[anomalies].plot()
# plt.show()
# plot_on_features(df_filtered_scaled_sample, pred_test)

# In[18]:
dl.l_plot(df_filtered_scaled_sample, pred_test)

# In[19]:
# Add anomalies to all columns from raw dataframe
# plot_on_features(dataset_raw, pred, anomay_label=-1) # todo: add list of columns to be ploted

# Anomaly detection method experiments

# In[20]:
# Define nine outlier detection tools to be compared
# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]
# Show all detectors
for i, clf in enumerate(dl.classifiers.keys()):
    print('Model', i + 1, clf)

# Decision Boundary plot of Isolation Forest
model_dir = os.getcwd() + "/models"
clf_name = 'Isolation Forest'
clf = dl.classifiers[clf_name]
filen = model_dir + '/' + 'docker_clf_' + type(clf).__name__ + '.cloudpickle'
pkl_IF = dl.read_data_file(
    filen, False, dl.build_model, 
    args={'clf_name': clf_name, 'clf': clf,
          'df': df_filtered_scaled,
          'df_testing': df_filtered_scaled_testing})

dl.count_anom(pkl_IF['y_pred_idx_test'], anom)

dl.l_plot(df_filtered_scaled, pkl_IF['y_pred_idx'])

dl.decision_boundary(pkl_IF['clf'],
                     df_filtered_scaled,
                     anomaly_label=1,
                     data_test=df_filtered_scaled_testing)

dl.decision_boundary2(pkl_IF, df_filtered_scaled, 
                      df_filtered_scaled_testing, 
                      legend_loc='upper right', size=200,
                      file="Decision_Boundary_{}.pdf".format(type(pkl_IF['clf']).__name__))

dl.decision_boundary3(pkl_IF, df_filtered_scaled, df_filtered_scaled_testing, size=30)

# Decision Boundary plot of CBLOF
clf_name = 'Cluster-based Local Outlier Factor (CBLOF)'
clf = dl.classifiers[clf_name]
filen = model_dir + '/' + 'docker_clf_' + type(clf).__name__ + '.cloudpickle'
pkl_CBLOF = dl.read_data_file(
    filen, False, dl.build_model, 
    args={'clf_name': clf_name, 'clf': clf,
          'df': df_filtered_scaled,
          'df_testing': df_filtered_scaled_testing})

dl.count_anom(pkl_CBLOF['y_pred_idx_test'], anom)

df_filtered_scaled.index.get_indexer(pkl_CBLOF['y_pred_idx'])
df_filtered_scaled.index[109]
pkl_CBLOF['y_pred_idx'][0]

dl.l_plot(df_filtered_scaled, pkl_CBLOF['y_pred_idx'])

dl.decision_boundary2(pkl_CBLOF, df_filtered_scaled, 
                      df_filtered_scaled_testing, 
                      legend_loc='upper right', size=200, title='CBLOF',
                      file="Decision_Boundary_{}.pdf".format(type(pkl_CBLOF['clf']).__name__))

# Decision Boundary plot of  VAE
clf_name = 'Variational auto encoder (VAE)'
clf = dl.classifiers[clf_name]
filen = model_dir + '/' + 'docker_clf_' + type(clf).__name__ + '.cloudpickle'
pkl_VAE = dl.read_data_file(
    filen, False, dl.build_model, 
    args={'clf_name': clf_name, 'clf': clf,
          'df': df_filtered_scaled,
          'df_testing': df_filtered_scaled_testing})

dl.count_anom(pkl_VAE['y_pred_idx_test'], anom)

dl.l_plot(df_filtered_scaled, pkl_VAE['y_pred_idx'])

dl.decision_boundary2(pkl_VAE, df_filtered_scaled, 
                      df_filtered_scaled_testing, 
                      legend_loc='upper right', size=200,
                      file="Decision_Boundary_{}.pdf".format(type(pkl_VAE['clf']).__name__))

#Preprocess dataset
# dataset_cmp = dataset_raw
print("="*15)
model_dir = "models"
# Fit the models with the GUIFI data and
classifier_stor = {}
for (clf_name, clf) in dl.classifiers.items():
    classifier_stor[clf_name] = dl.proc_classifier(
        clf_name, clf, df_filtered_scaled, df_filtered_scaled_testing, 
        fpref='test')

# In[22]:
# Saving results of experiments
exp_name = "dipet_exp_v5_2"
df_pred_training = pd.DataFrame.from_dict(predictions_training)
df_pred_testing = pd.DataFrame.from_dict(predictions_testing)
if os.path.isfile(os.path.join(model_dir,f'{exp_name}_training.csv')):
    print("Training exists ....")
else:
    df_pred_training.to_csv(os.path.join(model_dir,f'{exp_name}_training.csv'))

if os.path.isfile(os.path.join(model_dir,f'{exp_name}_testing.csv')):
    print("Testing exists ....")
else:
    df_pred_testing.to_csv(os.path.join(model_dir,f'{exp_name}_testing.csv'))

# In[23]:
df_pred_training

# In[24]:
df_pred_testing

# SHAP analysis
# In[25]:
df_filtered_scaled

# In[26]:
# Subsample the data for shap
# df_filtered_scaled_sample = df_filtered_scaled.sample(frac = 0.05)
df_filtered_scaled_sample = df_filtered_scaled_testing.loc["2021-04-14 01:55:00":"2021-04-14 23:50:00"] # exact anomaly frames
print(f"Sample size for shape analysis: {df_filtered_scaled_testing.shape}")
df_filtered_scaled_testing

# In[27]:
# %%capture output
# Isolation forest shap
clf_if5 = classifier_stor['Isolation Forest']
print("Start Shap explainer Isolation Forest...")

# shap_values
clf_if5

# In[30]:
sample_pred = clf_if5.predict(df_filtered_scaled_sample)

sample_anomaly_index = np.where(sample_pred==1)
sample_normal_index = np.where(sample_pred!=1)
print("Index of found anomalies")
print(sample_anomaly_index)
print("Normal index")
print(sample_normal_index)
print("Sample columns")
print(df_filtered_scaled_sample.columns)


# In[31]:
# Saving shape explainer instance and shap values
import pickle

#Shape explainer
load_shap_values = pickle.load(open('dipet_exp_v5_2_explainer_shap_3.sav', 'rb'))
shap_values_if5 = load_shap_values

# In[32]:
# shap_values[-1][18]
for e in sample_anomaly_index[0]:
    print(f"Annomaly loc {e}")
    shap.plots.force(shap_values_if5[e], matplotlib=True)

# In[33]:
# df_filtered_scaled_sample.values[18]

# In[46]:
plt.figure(figsize=(9, 9))
shap.plots.force(shap_values_if5[6])
# shap.force_plot(explainer.expected_value, shap_values[-1][18], df_filtered_scaled_sample.values[18], feature_names=df_filtered_scaled_sample.columns)


# In[47]:
#local feature importance plot, where the bars are the SHAP values for each feature.
shap.plots.waterfall(shap_values_if5[105])

# In[ ]:
shap.plots.waterfall(shap_values_if5[6])

# In[36]:
for e in sample_anomaly_index[0]:
    print(f"Annomaly loc {e}")
    shap.plots.waterfall(shap_values_if5[e])


# Each dot is a single prediction (row) from the dataset.
# The x-axis is the value of the feature (from the X matrix, stored in shap_values.data).
# The y-axis is the SHAP value for that feature (stored in shap_values.values), 
# which represents how much knowing that feature’s value changes the output of the model for that sample’s prediction. 
# The light grey area at the bottom of the plot is a histogram showing the distribution of data values.
# To show which feature may be driving these interaction effects we can color our feature dependence scatter plot 
# by another feature. If we pass the entire Explanation object to the color parameter then the scatter plot 
# attempts to pick out the feature column with the strongest interaction with our feature. 
# If an interaction effect is present between this other feature and the feature we are plotting it 
# will show up as a distinct vertical pattern of coloring. 

shap.plots.scatter(shap_values_if5[:,'txb.rate-41'], color=shap_values_if5)

# Explicitly control which feature is used for coloring, pass specific feature
shap.plots.scatter(shap_values_if5[:,'txb.rate-41'], color=shap_values_if5[:,"processes-3"])

# In[39]:
shap.summary_plot(shap_values_if5, df_filtered_scaled_sample, plot_type="violin")

# In[40]:
#The beeswarm plot is designed to display an information-dense summary of how the top features in a 
#dataset impact the model’s output. Each instance the given explanation is represented by a single 
#dot on each feature fow. The x position of the dot is determined by the SHAP value (shap_values.value[instance,feature]) of that feature, and dots “pile up” along each feature row to show density. 
#Color is used to display the original value of a feature (shap_values.data[instance,feature]). 
shap.plots.beeswarm(shap_values_if5)

# In[41]:
# heatmap plot function creates a plot with the instances on the x-axis, the model inputs on the y-axis, 
# and the SHAP values encoded on a color scale. By default the samples are ordered using shap.order.hclust, 
# which orders the samples based on a hierarchical clustering by their explanation similarity. 
# This results in samples that have the same model output for the same reason getting grouped together 
#(such as people with a high impact from capital gain in the plot below).
shap.plots.heatmap(shap_values_if5)

# In[42]:
#bar plot function creates a global feature importance plot where the global importance of each feature is taken to be the mean absolute value for that feature over all the given samples.
shap.plots.bar(shap_values_if5, max_display=30)

