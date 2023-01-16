# coding: utf-8
# functions used in dipet_exp_analysis.py
# Copyright (c) 2022  Gabriel Iuhasz and Llorenç Cerdà-Alabern
# script used to process the dataset in the paper "Anomaly
# Detection for Fault Detection in Wireless Community Networks Using
# Machine Learning"


import os
# limit GPU allocation
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
np.random.seed(42)
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import tqdm
import glob
import fnmatch
import gzip
import re
import sys
import random
import itertools
# Import all models
from joblib import dump, load
import sklearn as skl
from sklearn.decomposition import SparsePCA, PCA
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.feature_bagging import FeatureBagging
from pyod.models.iforest import IForest
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.alad import ALAD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.anogan import AnoGAN
from pyod.models.lunar import LUNAR
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
import time
import shap
import cloudpickle
import matplotlib
import matplotlib.font_manager as font_manager
matplotlib.rcParams.update({'font.size': 20})

import common as cmn

# pyod parameters
contamination_est = 0.005
# random_state = np.random.RandomState(42)
random_state  = 42
classifiers = {
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=contamination_est,
              check_estimator=False,
              n_clusters=8,
              alpha=0.9,
              beta=5,
              use_weights=False,
              random_state=random_state,
              n_jobs=-1),
    'Isolation Forest': IForest(n_estimators=20,
                                max_samples=0.7,
                                contamination=contamination_est,
                                random_state=random_state,
                                max_features=1.0,
                                bootstrap=False,
                                n_jobs=-1),
     'Auto Encoder (AE)':
     AutoEncoder(hidden_neurons=[64, 32, 32, 64], epochs=30, batch_size=32, dropout_rate=0.2,
                 contamination=contamination_est, verbose=1),
    'DeepSVDD (DeepSVDD)':
    DeepSVDD(use_ae=True,
             hidden_neurons=[128, 64, 32],
             hidden_activation='relu',
             output_activation='sigmoid',
             optimizer='adam',
             epochs=20,
             batch_size=64,
             dropout_rate=0.3,
             l2_regularizer=0.1,
             validation_size=0.3,
             preprocessing=True,
             random_state=random_state,
             contamination=contamination_est),
    'DeepSVDD_nonae(60epochs)':
    DeepSVDD(use_ae=False,
             hidden_neurons=[128, 64, 32],
             hidden_activation='relu',
             output_activation='sigmoid',
             optimizer='adam',
             epochs=60,
             batch_size=64,
             dropout_rate=0.2,
             l2_regularizer=0.1,
             validation_size=0.1,
             preprocessing=True,
             random_state=random_state,
             contamination=contamination_est),
    'Variational auto encoder (VAE) [opacity09_gamma05]':
    VAE(encoder_neurons=[128, 64, 32],
        decoder_neurons=[32, 64, 128],
        hidden_activation='relu',
        # loss=keras.losses.mean_squared_error
        gamma=0.5,
        capacity=0.9,
        epochs=30,
        batch_size=32,
        dropout_rate=0.2,
        l2_regularizer=0.1,
        contamination=contamination_est,
        verbose=1),
    'Adversarially Learned Anomaly Detection (ALAD)':
    ALAD(
        activation_hidden_gen='tanh',
        activation_hidden_disc='tanh',
        output_activation=None,
        dropout_rate=0.2,
        latent_dim=2,
        dec_layers=[5, 10, 25],
        enc_layers=[25, 10, 5],
        disc_xx_layers=[25, 10, 5],
        disc_zz_layers=[25, 10, 5],
        disc_xz_layers=[25, 10, 5],
        learning_rate_gen=0.0001,
        learning_rate_disc=0.0001,
        add_recon_loss=False,
        lambda_recon_loss=0.1,
        epochs=200,
        verbose=0,
        preprocessing=True,
        add_disc_zz_loss=True,
        spectral_normalization=False,
        batch_size=32,
        contamination=contamination_est
    ),
    'Anomaly Detection with Generative Adversarial networks (AnoGAN)':
    AnoGAN(
        activation_hidden='tanh',
        dropout_rate=0.2,
        latent_dim_G=2,
        G_layers=[20, 10, 3, 10, 20],
        verbose=0,
        D_layers=[20, 10, 5],
        index_D_layer_for_recon_error=1,
        epochs=500,
        preprocessing=False,
        learning_rate=0.001,
        learning_rate_query=0.01,
        epochs_query=20,
        batch_size=32,
        output_activation=None,
        contamination=contamination_est
    ),
    'Unifying Local Outlier Detection via Graph Neural Networks (LUNAR)':
        LUNAR(
            model_type='WEIGHT',
            n_neighbours=5,
            negative_sampling='MIXED',
            val_size=0.1,
            scaler=MinMaxScaler(),
            epsilon=0.1,
            proportion=1.0,
            n_epochs=200,
            lr=0.001,
            wd=0.1,
            verbose=1
        ),
    'Single-Objective Generative Adversarial Active Learning (SO-GAAL)':
        SO_GAAL(
            stop_epochs=30,
            lr_d=0.01,
            lr_g=0.0001,
            momentum=0.6,
            contamination=contamination_est
        ),
    'Multi-Objective Generative Adversarial Active Learning (MO-GAAL)':
    MO_GAAL(
        k=20,
        stop_epochs=30,
        lr_d=0.01,
        lr_g=0.0001,
        momentum=0.9,
        contamination=contamination_est
    )

}

if 'VAE' in sys.modules: 
    classifiers.update({
        'Variational auto encoder (VAE)':
        VAE(encoder_neurons=[128, 64, 32], decoder_neurons=[32, 64, 128], 
            epochs=30, batch_size=32,
            dropout_rate=0.2,contamination=contamination_est, verbose=1)})
else:
    print("skipping VAE classifier, library not installed")

# In[3]:
def save_figure(file):
    plt.savefig('figures/'+file, format='pdf',
                bbox_inches='tight', pad_inches=0)

def filter_low_variance(df):
    print("Checking low variance columns ...")
    print("Starting Clean dataset")
    uniques = df.apply(lambda x: x.nunique())
    rm_columns = []
    for uindex, uvalue in uniques.iteritems():
        if uvalue == 1:
            rm_columns.append(uindex)
    print(f"First pass columns to be removed {len(rm_columns)}")
    print(rm_columns)
    print(f"Initial shape: {df.shape}")
    df_filtered = df.drop(rm_columns, inplace=False, axis=1)
    print(f"Filtered shape: {df_filtered.shape}")
    return df_filtered, rm_columns

def add_agg_traffic(df):
    for f,e,w in ((f, 'eth.'+f, 'wifi.'+f) for f in (
            f+'.rate' for f  in ['txe', 'rxe', 'txb', 'rxb', 'txp', 'rxp'])):
        if e in df and w in df:
            df[f] = df[e].fillna(0) + df[w].fillna(0)
        elif e in df:
            df[f] = df[e].fillna(0)
        elif w in df:
            df[f] = df[w].fillna(0)
    for f in ['b.rate', 'p.rate']:
        if 'rx'+f in df and 'tx'+f in df:
            df['sum.x'+f]  = df['rx'+f] + df['tx'+f]
            df['diff.x'+f] = df['rx'+f] - df['tx'+f]
        else:
            return False
    return True

def get_traffic_features(df):
    tf = []
    for f in [re.search(r'^(.*(?:xb.rate|xe.rate|xp.rate).*)$', f) 
              for f in df.columns]:
        if f:
            tf.append(f.group(1))
    return tf

def get_non_traffic_features(df):
    return set(df.columns).difference(set(get_traffic_features(df)))

def load_hd_data(hdf):
    f = gzip.open(hdf)
    df = pd.read_csv(f, index_col=0)
    df.index = pd.to_datetime(df.index)
    return(df)

def read_dataset(date=None, node='*', only_traffic=True, add_traffic=True,
                 per_node=False, dropcol=['name', 'uid', 'id']):
    """ returns a DataFrame with node's features
    date: dataset file month list (None for all months)
    node: node file wildcard to read
    only_traffic: read only traffic features
    add_traffic: add features with aggregated traffic
    per_node: False return a single DataFrame with all features,
              True returns a dic with per node features
    """
    # cmn = importlib.import_module("common")
    dataset = {} if per_node else pd.DataFrame()
    # build a list of rate features
    ratef = set([f for f in [x+y+'.rate' for x in ['rx', 'tx'] 
                             for y in ['b', 'p']]] + 
                [f+p for f in ('eth.', 'wifi.') for p in (
                    f+'.rate' for f in ['txe', 'rxe', 'txb', 'rxb', 'txp', 'rxp'])])
    if not date:  # read dates from files
        date = list({m.group(1) for m in
                     [re.search(r'^(\d\d-\d\d)-state-\d+\.csv.gz', f) for f in
                      fnmatch.filter(os.listdir(csvdir), '*-state-*csv.gz')]})
        date = sorted(date, reverse=False)
    for d in date:
        fpat = d+'-state-'+node+'.csv.gz'
        if not per_node:
            tmpdf = pd.DataFrame()
        print("Processing "+d)
        file_list = fnmatch.filter(os.listdir(csvdir), fpat)
        if len(file_list) == 0:
            print(csvdir+fpat+'?')
            continue
        for f in file_list:
            m = re.search(r'-state-(\d+)\.csv.gz', f)
            n = int(m.group(1)) if m else print("node? "+f)
            hd = load_hd_data(csvdir + f)
            if dropcol:
                hd.drop(dropcol, axis=1, inplace=True)
            if only_traffic:
                hd.drop(list(set(hd.columns) - ratef), axis=1, inplace=True)
            if not add_agg_traffic(hd):
                print("empty? "+f)
                continue
            hd.columns = [f+'-'+str(n) for f in hd.keys()]
            if per_node:
                dataset.update({n: hd})
            else:
                if len(tmpdf) == 0:
                    tmpdf = hd.copy()
                else:
                    tmpdf = tmpdf.join(hd, how='outer')
        if not per_node:
            if len(dataset) == 0:
                dataset = tmpdf.copy()
            else:
                dataset = dataset.append(tmpdf, sort=False)
    return dataset

def load_filter_low_variance(data_loc,
                             reg_file,
                             low_varaince=True,
                             dropcol=['name', 'uid', 'id',]):
    """
    :param data_loc: directory where files are located
    :param reg_file: name of the csv to be loaded, if * at end only specified node is considered,
    if * at the beginning specifies time
    :param dropcol: list of columns to drop from all df
    :return: dataframe
    """
    print("Data location: ")
    print(data_loc)
    print("Work dir")
    print(os.getcwd())
    print("Checking files in data location ...")

    # print(reg_file.split('.')[0][-1])
    if reg_file.split('.')[0][-1] == "*":
        same_node = False
    else:
        same_node = True

    files = glob.glob(os.path.join(data_loc, reg_file))
    # print(files)
    # sys.exit()
    if same_node:
        # Load files for a particular node based on reg_file

        list_df = []
        for f in tqdm.tqdm(files):
            # print(f)
            df = pd.read_csv(f)
            df.drop(dropcol, inplace=True, axis=1)
            add_agg_traffic(df)
            list_df.append(df)

        # Get Column file
        column_size = []
        for d in list_df:
            # print(d.shape)
            column_size.append(d.shape[-1])
        if low_varaince:
            from collections import Counter
            items = Counter(column_size).keys()
            print("Unique column size: {}".format(items))
            print(f"Columns found: {list_df[0].columns}")

            df = list_df[0]
            print("Checking low variance columns ...")
            print("Starting Clean dataset")
            uniques = df.apply(lambda x: x.nunique())
            rm_columns = []
            for uindex, uvalue in uniques.iteritems():
                if uvalue == 1:
                    rm_columns.append(uindex)
            print("First pass columns to be removed")
            print(rm_columns)

            # Find common low variance elements in list of low variance columns
            list_low_varaince = []
            for df in list_df:
            # Check for monotonous columns and remove them
                print("Checking low variance columns ...")
                print("Starting Clean dataset")
                uniques = df.apply(lambda x: x.nunique())
                rm_columns = []
                for uindex, uvalue in uniques.iteritems():
                    if uvalue == 1:
                        rm_columns.append(uindex)
                print(rm_columns)
                list_low_varaince.append(rm_columns)

            common_low_variance = list(set.intersection(*map(set, list_low_varaince)))
            print("Common low variance columns:")
            print(common_low_variance)

            # Remove common low variance columns
            for df in list_df:
                df.drop(common_low_variance, inplace=True, axis=1)

        # Concat all dataframes
        df_concat = pd.concat(list_df, ignore_index=True)
        print(f"Concatenated all data, new shape is {df_concat.shape} and columns are:")
        print(df_concat.shape)
        return df_concat
    else:
        dict_all_df = {}
        for e in files:
            node_id = e.split("/")[-1].split('.')[0].split('-')[-1] # add node values
            df = pd.read_csv(e, index_col=0)
            # df = pd.read_csv(f, index_col=0)
            df.index = pd.to_datetime(df.index)
            # drop specified columns
            df.drop(dropcol, inplace=True, axis=1)
            add_agg_traffic(df)
            if node_id not in dict_all_df.keys():
                dict_all_df[node_id] = [df]
            else:
                dict_all_df[node_id].append(df)

        # print(dict_all_df)
        print("Consolidating individual dataframe ...")
        dict_all_concat_df = {}
        for k, v in tqdm.tqdm(dict_all_df.items()): # todo: fix node 6 issues
            # Concat on rows
            df_concat = pd.concat(v, ignore_index=False)
            # df_filtered = df_concat.filter(like="rate", axis=1)
            # Save dataframe
            # df_concat.add_suffix(f'_{k}').to_csv(os.path.join(concat_dir, f'state_{k}.csv'), index=False)
            dict_all_concat_df[k] = df_concat.add_suffix(f'-{k}')
        # print(dict_all_concat_df['7'])
        df_all = pd.DataFrame()

        print("Consolidating all columns ...")
        for k,v in tqdm.tqdm(dict_all_concat_df.items()):
            # filter based on columns containing rate
            # df_filtered = v.filter(like="rate", axis=1)
            df_all = pd.concat([df_all, v], axis=1)
        if low_varaince:
            print("Checking low variance columns ...")
            print("Starting Clean dataset")
            uniques = df_all.apply(lambda x: x.nunique())
            rm_columns = []
            for uindex, uvalue in uniques.iteritems():
                if uvalue == 1:
                    rm_columns.append(uindex)
            print(f"First pass columns to be removed: {len(rm_columns)}")

            df_all.drop(rm_columns, inplace=True, axis=1)
        return df_all

def filter_columns_wildcard(df, wild_card, keep=True):
    """
    :param df: dataframe to filer
    :param wild_card: str wildcard of columns to be filtered
    :param keep: if keep True, only cols with wildcard are kept, of False they will be deleted
    :return: filtered dataframe
    """
    filter_col_wildcard = wild_card
    filtr_list = []
    mask = df.columns.str.contains(filter_col_wildcard)
    filtr_list.extend(list(df.loc[:,mask].columns.values))
    print("Columns to be filtered:")
    print(filtr_list)
    if keep:
        df_concat_filtered = df[filtr_list]
    else:
        df_concat_filtered = df.drop(filtr_list, axis=1)

    print(f"Filtered shape: {df_concat_filtered.shape}")
    print("Columns of filtered data:")
    print(df_concat_filtered.columns)
    return df_concat_filtered

def filter_nan_col(df_concat_filtered,
                   drop_threashold=0.4,
                   fillna=True,
                   verbose=0):
    """
    :param df_concat_filtered: dataframe to be processed
    :param drop_threashold: threashold for dropping columns based on NaN values
    :param fillna: Fill NaN with 0
    """
    # Count nan values per column and set drop threashold
    drop_list_th = []
    for c in df_concat_filtered.columns.values:
        percentage_nan = 100*df_concat_filtered[c].isna().sum()/\
            len(df_concat_filtered)
        if verbose:
            print(f"Length of column {c} is {len(df_concat_filtered)}")
            print(f"Of which {df_concat_filtered[c].isna().sum()} are nan")
            print(f"Percentage of nan in {c} is {percentage_nan/100}")
        if percentage_nan/100 > drop_threashold:
            print(f"Percentage of nan in {c} is {percentage_nan}")
            drop_list_th.append(c)
    print(f"Colsumns before nan drop: {df_concat_filtered.shape[-1]}")
    print(f"Columns to be droped {len(drop_list_th)}")
    # print(df_concat_filtered.columns)
    if len(drop_list_th) > 0:
        print(drop_list_th)
        df_concat_filtered.drop(drop_list_th, axis=1, inplace=True)
    if fillna:
        df_concat_filtered.fillna(value=0, inplace=True)

def plot_on_features(data, pred, features=[], anomay_label=-1):
    """
    :param data: dataset used for training or prediction
    :param pred: model prediction
    :param anomay_label: label for anomaly instances (differs from method to method)
    :return:
    """
    if not features:
        col_names_plt = list(data.columns.values)
    else:
        col_names_plt = features
    data['anomaly'] = pred
    for feature in col_names_plt:
        if feature == 'epoch' or feature == 'anomaly':
            pass
        else:
            # fig, ax = plt.subplots(figsize=(15,10))
            a = data[data['anomaly'] == anomay_label] #anomaly
            _ = plt.figure(figsize=(18,6))
            _ = plt.plot(data[feature], color='blue', label='Normal')
            _ = plt.plot(a[feature], linestyle='none', marker='X', color='red',
                         markersize=4, label='Anomaly')
            _ = plt.xlabel('Date and Time')
            _ = plt.ylabel(f'{feature}')
            _ = plt.title(f'{feature} Anomalies')
            _ = plt.grid()
            _ = plt.legend(loc='best')
            plt.show();

def l_plot(df,
           anom_idx,
           anomaly_label = -1,
           title=0):
    dfs = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 8})
    sum_xb_rate = []
    for f in df.columns:
        if(re.search(r'sum.xb.rate', f)):
            sum_xb_rate.append(f)
    traffic_var = pd.DataFrame(
        df[sum_xb_rate].var(axis=0), 
        columns=['var']).sort_values(by='var', ascending=False)
    # plot features
    res = df[traffic_var.index[range(32)]]
    print(f"Number of anom_idx: {len(anom_idx)}")
    # fig = plt.figure(figsize=(20,20))
    fig, ax = plt.subplots(nrows=16, ncols=2, sharex='col', figsize=(9,9))
    ax = res.plot(ax=ax, subplots=True, 
                  # markevery=list(anomaly_index_df), 
                  # marker="o", 
                  # marker=r'$\circ$',
                  mfc = "blue", fontsize=8, rot=45)
    # for a in ax:
    #     a.legend(prop={'size': 8})
    # mark anom_idx
    #     print(res.index)
    #     print(res.loc[anom_idx])
    res.loc[anom_idx].plot(
        ax=ax, subplots=True, ls="none",
        marker="o", markersize=3, fontsize=8, 
        fillstyle='none',
        color='red', rot=45,
        legend=False, xlabel='Date', ylabel='Mbps')
    # if title:
    #     res.loc[anom_idx].plot(ax=ax, subplots=True, ls="none", marker="o",
    #                            fontsize=8, rot=45, legend=False,
    #                            title=title)
    # else:
    #     res.loc[anom_idx].plot(ax=ax, subplots=True, ls="none", marker="o",
    #                            color='blue', fontsize=8, rot=45, legend=False)
    plt.show()
    matplotlib.rcParams.update({'font.size': dfs})

def transform_df(model, df):
    res = pd.DataFrame(model.transform(df), index=df.index)
    if res.shape[1] == df.shape[1]:
        res.columns = df.columns
    return res

def d_statistic(t, mua, sda):
    return sum(((t.values-mua.values)/sda.values)**2)

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
class PCAclf(BaseEstimator):
    pca = None
    pca_comp = None
    Qn = None
    QnQ = None
    Dn = None
    DnQ = None
    mua = None
    sda = None
    fitted = False
    def get_score(self, df):
        if type(df).__name__ == 'DataFrame':
            score = pd.DataFrame(self.pca.transform(df),
                                 index=df.index)
            score_inv = pd.DataFrame(self.pca.inverse_transform(score),
                                     index=df.index,
                                     columns=df.columns)
        else:
            score = self.pca.transform(df)
            score_inv = self.pca.inverse_transform(score)
        return (score, score_inv)
    def get_residuals(self, df):
        score, score_inv = self.get_score(df)
        return df - score_inv
    def get_Q(self, df):
        residuals = self.get_residuals(df)
        if type(df).__name__ == 'DataFrame':
            return residuals.apply(lambda x: sum(x**2), axis=1)
        return np.apply_along_axis(lambda x: sum(x**2), 1, residuals)
    def fit(self, data, pca_comp=None, seed=42, var=0.95, contamination_est=0.005):
        if self.fitted:
            print("PCA already fitted. pca_comp={}".format(self.pca_comp))
            return
        self.fitted = True
        df = data.copy()
        df = df.fillna(0)
        if pca_comp:
            self.pca_comp = pca_comp
        else:
            np.random.seed(seed)
            pca = skl.decomposition.PCA(svd_solver='auto')
            pca.fit(df)
            cs_tf = np.cumsum(pca.explained_variance_ratio_)
            self.pca_comp = np.argmax(cs_tf >= var)+1
        np.random.seed(seed)
        print("Fitting. pca_comp={}".format(self.pca_comp))
        self.pca = skl.decomposition.PCA(n_components=self.pca_comp, svd_solver='auto')
        self.pca.fit(df)
        score, score_inv = self.get_score(df)
        residuals = score - score_inv
        self.Qn = residuals.apply(lambda x: sum(x**2), axis=1)
        self.QnQ = self.Qn.quantile(1-contamination_est)
        self.mua = score.mean(axis=0)
        self.sda = score.std(axis=0)
        self.Dn = score.apply(d_statistic, axis=1, args=(self.mua, self.sda, ))
        self.DnQ = self.Dn.quantile(1-contamination_est)
    def decision_function(self, df):
        Qn = self.get_Q(df)
        return Qn > self.QnQ
    def predict(self, df):
        return self.decision_function(df)
    def decision_function_D(self, df):
        Dn = self.get_D(df)
        return Dn > self.DnQ
    def predict_D(self, df):
        return self.decision_function_D(df)
    def cdc_Q(self, x):
        CDC = (np.array(x) - 
           self.pca.inverse_transform(
               self.pca.transform(np.array(x).reshape(1, -1))))**2
        return pd.Series(data=CDC[0,:], index=x.index, name='CDCq')
    def cdc_D(self, x):
        P = self.pca.components_.T
        CDC = (P @ np.diag(1/np.sqrt(self.pca.explained_variance_ratio_)) @ P.T @ x)**2
        return pd.Series(data=CDC, index=x.index, name='CDCd')

class MeshmonTransformer:
    type_ = None
    scaler = None
    non_traffic_maxmin = {}
    traffic_weight = None
    def fit(self, df, type_):
        """
        type = {'minmax', 'max','meanmax'}: traffic transformer type
        """
        self.type_ = type_
        if self.type_ == 'minmax':
            self.scaler =  MinMaxScaler()
            self.scaler.fit(df)
        else:
            self.non_traffic_maxmin = {
                c: (np.min(df[c]), np.max(df[c]))
                for c in get_non_traffic_features(df)}
            self.traffic_weight = {
                'meanmax': np.mean(df[get_traffic_features(df)].max()),
                'max': np.max(df[get_traffic_features(df)].max())
            }[self.type_]
    def transform(self, df):
        if self.type_ == 'minmax':
            return self.scaler.transform(df)
        # 'max','meanmax'
        traffic_col = list(set(df.columns).difference(
            set(self.non_traffic_maxmin.keys())))
        # make a copy of dataframe
        scaled_features = df.copy()
        # scale non traffic features
        for c in self.non_traffic_maxmin.keys():
            if (self.non_traffic_maxmin[c][1] > self.non_traffic_maxmin[c][0]):
                scaled_features[c] = (scaled_features[c]-
                                      self.non_traffic_maxmin[c][0])/(
                                          self.non_traffic_maxmin[c][1]-
                                          self.non_traffic_maxmin[c][0])
            else:
                scaled_features[c] = (scaled_features[c]-
                                      self.non_traffic_maxmin[c][0])
        # scale traffic features
        scaled_features[traffic_col] = (scaled_features[traffic_col]/
                                            self.traffic_weight)
        return scaled_features

def count_anom(anom, vline):
    if type(anom).__name__ == 'DataFrame':
        anom = anom.index
    return (
        len(anom),
        sum([(np.datetime64(f) > np.datetime64(vline[0])) &
                (np.datetime64(f) < np.datetime64(vline[1]))
                for f in anom]))

def train_model_and_count_anom(clf_name, training, testing, anom):
    model = build_model(clf_name=clf_name, 
                        clf=classifiers[clf_name],
                        df=training, 
                        df_testing=testing)
    return (count_anom(model['y_pred_idx_test'], anom) +
            (model['stop_training_time'],
             model['stop_prediction_time']))

def decision_boundary(model,
                      data,
                      anomaly_label=-1,
                      contamintation=0.005,
                      data_test=None
                     ):
    """
    :param model: model to be refitted with 2 features (PCA)
    :param data: dataset
    :param anomaly_label: label for anomaly instances (differs from method to method)
    """
    method = type(model).__name__
    transformer = PCA(n_components=2)
    transformer.fit(data)
    data = transform_df(transformer, data)
    # print("PCA data shape: {}".format(data.shape))
    # fit model
    model_clone = clone(model)
    try:
         # becouse we have only two features we must override previous setting
        model_clone.set_params(
            max_features=data.shape[-1]) 
    except ValueError:
        try:
            model_clone.set_params(
                encoder_neurons=[2, 64, 32],
                decoder_neurons=[32, 64, 2]
            )
            print("DNN detected")
        except ValueError:
            model_clone = model
            pass
    model_clone.fit(data)
    y_pred_outliers = model_clone.predict(data) # added anomaly_label for testing
    # predict raw anomaly score
    y_pred_scores = model_clone.decision_function(data) *-1# added for testing
    print(f"Decision function out: {y_pred_scores}")
    # Threshold val for identifying outs and ins 
    threshold = np.percentile(y_pred_scores, 100 * contamintation) # added for testing
    print(threshold)
    # get anomaly index
    anomaly_index_rf = np.where(y_pred_outliers == anomaly_label)
    # Get anomalies based on index
    ano_rf = data.iloc[anomaly_index_rf]
    # plot the line, the samples, and the nearest vectors to the plane
    #     xx, yy = np.meshgrid(np.linspace(-15, 25, 80), np.linspace(-5, 20, 80))
    # Set size of the mesh
    h = 100
    x_min, x_max = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1
    y_min, y_max = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1
    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))
    Z = model_clone.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    # plt.figure(figsize=(10,10),  dpi = 600)
    plt.figure(figsize=(9,9))
    plt.title(f"Decision Boundary for {method}")
    plt.contourf(xx, yy, Z, 
                 # levels=np.linspace(Z.min(),      #added for testing
                 #                   threshold, 7), #added for testing
                 cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[
        threshold # changed for testing from 0
    ], linewidths=2, colors='black')
    #     plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
    b1 = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c='white',
                     s=20, edgecolor='k')
    c = plt.scatter(ano_rf.iloc[:, 0], ano_rf.iloc[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    if type(data_test).__name__ == 'DataFrame':
        data_test = transformer.transform(data_test)
        y_pred_outliers_test = model_clone.predict(data_test)
        # import pdb; pdb.set_trace()
        anomaly_index_rf_test = np.where(y_pred_outliers_test == anomaly_label)
        ano_rf_test = data_test[anomaly_index_rf_test]
        d = plt.scatter(ano_rf_test[:, 0], ano_rf_test[:, 1], c='yellow',
                        s=20, edgecolor='k')
        plt.legend([a.collections[0], b1, c, d],
                   ["learned decision function",
                    "normal",
                    "anomaly", "testing"],
                   loc="upper left", 
                   prop=font_manager.FontProperties(style='normal', size=16))        
    else:
        plt.legend([a.collections[0], b1, c],
                   ["learned decision function",
                    "normal",
                    "anomaly", ],
                   loc="upper left", 
                   prop=font_manager.FontProperties(style='normal', size=16))
    plot_name = f"Decision_Boundary_{method}.pdf"
    # plt.savefig(os.path.join(paper_data_folder, plot_name),  format='pdf',
    #        bbox_inches='tight', pad_inches=0)
    plt.show()
    # plt.close();

# projection of anomalies in 2D PCA
def decision_boundary2(pkl, data, data_test, contamintation=0.005,
                       legend_loc='upper left', size=20, font=20, 
                       title=None, file=None):
    """
    :param pkl: pickle file
    :param data: dataset
    :param data_test: testing set
    """
    method = type(pkl['clf']).__name__
    if not title:
        title = pkl['clf_name']
    transformer = PCA(n_components=2)
    transformer.fit(data)
    data2 = transform_df(transformer, data)
    anom2 = transform_df(transformer, data.loc[pkl['y_pred_idx']])
    normal = data2.loc[list(set(data2.index).difference(set(anom2.index)))]
    anom3 = transform_df(transformer, data_test.loc[pkl['y_pred_idx_test']])
    h = 100
    x_min, x_max = data2.iloc[:, 0].min() - 1, data2.iloc[:, 0].max() + 1
    y_min, y_max = data2.iloc[:, 1].min() - 1, data2.iloc[:, 1].max() + 1
    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))
    # because we have only two features we must override previous setting
    model_clone = clone(pkl['clf'])
    if method == 'VAE':
        model_clone.set_params(
            encoder_neurons=[2, 64, 32],
            decoder_neurons=[32, 64, 2]
        )
        print("DNN detected")
    elif method == 'IForest':
        model_clone.set_params(
            max_features=data2.shape[-1]) 
    model_clone.fit(data2)
    Z = model_clone.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    # import pdb; pdb.set_trace()
    # predict raw anomaly score
    y_pred_scores = model_clone.decision_function(data2) *-1# added for testing
    print(f"Decision function out: {y_pred_scores}")
    # Threshold val for identifying outs and ins 
    threshold = np.percentile(y_pred_scores, 100 * contamintation) # added for testing
    print(threshold)
    # plt.figure(figsize=(10,10),  dpi = 600)
    plt.figure(figsize=(9,9))
    plt.title(f"{title}")
    plt.contourf(xx, yy, Z, 
                 # levels=np.linspace(Z.min(),      #added for testing
                 #                   threshold, 7), #added for testing
                 cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[
        threshold # changed for testing from 0
    ], linewidths=2, colors='black')
    #     plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
    b1 = plt.scatter(normal.iloc[:, 0], normal.iloc[:, 1], c='white',
                     s=size, edgecolor='k') # r'$\circ$')
    c = plt.scatter(anom2.iloc[:, 0], anom2.iloc[:, 1], c='yellow',
                    s=size+5, edgecolor='k', marker="s")
    d = plt.scatter(anom3.iloc[:, 0], anom3.iloc[:, 1], c='red',
                    s=2*size, edgecolor='k', marker="*")
    plt.axis('tight')
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.ylabel('PC1')
    plt.xlabel('PC2')
    plt.legend([a.collections[0], b1, c, d],
               ["decision func.",
                "normal",
                "train. anom.", "test. anom." ],
               loc=legend_loc, labelspacing=0,
               prop=font_manager.FontProperties(style='normal', size=font))
    # plot_name = f"Decision_Boundary_{method}.pdf"
    # plt.savefig(os.path.join(paper_data_folder, plot_name),  format='pdf',
    #        bbox_inches='tight', pad_inches=0)
    plt.show()
    # plt.close();
    if file:
        save_figure(file)

# projection of anomalies in 2D PCA, no decision boundaries
def decision_boundary3(pkl, data, data_test, contamintation=0.005, size=20):
    """
    :param pkl: pickle file
    :param data: dataset
    :param data_test: testing set
    """
    method = type(pkl['clf']).__name__
    transformer = PCA(n_components=2)
    transformer.fit(data)
    # import pdb; pdb.set_trace()
    data2 = transform_df(transformer, data)
    anom2 = transform_df(transformer, data.loc[pkl['y_pred_idx']])
    anom3 = transform_df(transformer, data_test.loc[pkl['y_pred_idx_test']])
    # Z = pkl['scores_pred']
    # threshold = np.percentile(Z, 100 * contamintation) # added for testing
    # print(threshold)
    # predict raw anomaly score
    plt.figure(figsize=(9,9))
    plt.title(f"Decision Boundary for {method}")
    # plt.contourf(data2.iloc[:,0], data2.iloc[:,1], Z, 
    #              # levels=np.linspace(Z.min(),      #added for testing
    #              #                   threshold, 7), #added for testing
    #              cmap=plt.cm.Blues_r)
    # a = plt.contour(data2.iloc[:,0], data2.iloc[:,1], Z, levels=[
    #     threshold # changed for testing from 0
    # ], linewidths=2, colors='black')
    #     plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
    b1 = plt.scatter(data2.iloc[:, 0], data2.iloc[:, 1], c='white',
                     s=size, edgecolor='k')
    c = plt.scatter(anom2.iloc[:, 0], anom2.iloc[:, 1], c='red',
                    s=size, edgecolor='k')
    d = plt.scatter(anom3.iloc[:, 0], anom3.iloc[:, 1], c='yellow',
                    s=size, edgecolor='k')
    plt.axis('tight')
    plt.legend([b1, c, d],
               ["normal",
                "anomaly", "testing"],
               loc="upper left", 
               prop=font_manager.FontProperties(style='normal', size=16))
    # plot_name = f"Decision_Boundary_{method}.pdf"
    # plt.savefig(os.path.join(paper_data_folder, plot_name),  format='pdf',
    #        bbox_inches='tight', pad_inches=0)
    plt.show()
    # plt.close();

# In[21]:
# https://github.com/yzhao062/pyod/issues/256
# https://stackoverflow.com/questions/41754247/keras-save-model-issue
# https://keras.io/guides/serialization_and_saving/
# https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model
def read_data_file(fname, force, callf, args):
    if not force and os.path.isfile(fname): # and not  'VAE' in str(fname):
        cmn.say("reading file: " + fname)
        with open(fname, 'rb') as filehandle:
            if 'VAE' in str(fname):
                res = cloudpickle.load(filehandle)
                # res['clf'].model_= load_model(os.path.splitext(fname)[0])
                res['clf'].model_ = res['clf']._build_model()
            else:
                res = cloudpickle.load(filehandle)
    else:
        cmn.error("building file: " + fname)
        res = callf(**args)
        if res != None: # and not 'VAE' in str(fname):
            with open(fname, 'wb') as filehandle:
                if 'VAE' in str(fname):
                    # res['clf'].model_.save(
                    #     filepath=os.path.splitext(fname)[0],
                    #     include_optimizer=False,
                    #     save_format='tf',
                    #     save_traces=False)
                    model = res['clf'].model_
                    res['clf'].model_ = None
                    cloudpickle.dump(res, filehandle)
                    res['clf'].model_= model
                else:
                    cloudpickle.dump(res, filehandle)
    return res

def build_model(clf_name, clf, df, df_testing):
    start_training_time = time.time()
    print('fitting', clf_name)
    clf.fit(df)
    stop_training_time = time.time() - start_training_time
    print("Completed {} training in:  {}".format(clf_name, stop_training_time))
    scores_pred = clf.decision_function(df) * -1
    scores_pred_test = clf.decision_function(df_testing) * -1
    # predictions_training[f'{clf_name} df'] = scores_pred
    start_prediciton_time = time.time()
    y_pred_n = clf.predict(df)
    y_pred_idx = df.index[y_pred_n==1]
    y_pred_test = clf.predict(df_testing)
    y_pred_idx_test = df_testing.index[y_pred_test==1]
    stop_prediction_time = time.time() - start_prediciton_time
    print("Completed {} prediction in:  {}".format(clf_name, stop_prediction_time))
    print("Training decision function ...")
    return {'stop_training_time': stop_training_time,
            'stop_prediction_time': stop_prediction_time, 
            'clf_name': clf_name, 'clf': clf,
            'y_pred_idx': y_pred_idx,
            'y_pred_idx_test': y_pred_idx_test,
            'scores_pred': scores_pred,
            'scores_pred_test': scores_pred_test}

model_dir = "models"
anomaly_label = 1

def proc_classifier(clf_name, clf,
                    df_filtered_scaled,
                    df_filtered_scaled_testing, fpref='docker_clf'):
    filen = model_dir + '/' + fpref + '_' + type(clf).__name__ + '.cloudpickle'
    pkl = read_data_file(filen, False, build_model, 
                         args={'clf_name': clf_name, 'clf': clf,
                               'df': df_filtered_scaled,
                               'df_testing': df_filtered_scaled_testing})
    if 'stop_prediction_time' in pkl:
        print("Completed {} prediction in:  {}".format(
            clf_name, pkl['stop_prediction_time']))
    transformer = PCA(n_components=2)
    transformer.fit(df_filtered_scaled)
    X_transformed = transform_df(transformer, df_filtered_scaled)
    print("Initial shape: {}".format(df_filtered_scaled.shape))
    print("PCA data shape: {}".format(X_transformed.shape))
    print("Anomalies in dataset of size {} found: {}".format(
        len(df_filtered_scaled),len(pkl['y_pred_idx'])))
    # Plot prediction on 2d
    fig = plt.figure(figsize=(9,9))
    plt.scatter(X_transformed.iloc[:,0], X_transformed.iloc[:,1])
    plt.scatter(X_transformed.loc[pkl['y_pred_idx']].iloc[:,0], 
                X_transformed.loc[pkl['y_pred_idx']].iloc[:,1], color='r', marker='x')
    plt.legend()
    plt.grid()
    plt.title("{} plot with a total of {} anomalies".format(
        clf_name, len(pkl['y_pred_idx'])))
    plt.show()
    # Data preprocessing options (3d)
    transformer_3d = PCA(n_components=3)
    transformer_3d.fit(df_filtered_scaled)
    X_transformed_3d_n = transform_df(transformer_3d, df_filtered_scaled)
    print(X_transformed_3d_n.shape)
    # Plot 3D anomaly map
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel("x_composite_3")
    # Plot the compressed data points
    ax.scatter(X_transformed_3d_n.iloc[:, 0], X_transformed_3d_n.iloc[:, 1],
               zs=X_transformed_3d_n.iloc[:, 2], s=4, lw=1, label="normal",c="blue")
    # Plot x's for the ground truth outliers
    ax.scatter(X_transformed_3d_n.loc[pkl['y_pred_idx']].iloc[:,0],
               X_transformed_3d_n.loc[pkl['y_pred_idx']].iloc[:,1],
               X_transformed_3d_n.loc[pkl['y_pred_idx']].iloc[:,2],
               lw=2, s=60, marker="x", c="red", label="anomalies")
    ax.legend()
    plt.title("{} 3D plot with a total of {} anomalies".format(
        clf_name, len(pkl['y_pred_idx'])))
    plt.show()
    l_plot(df=df_filtered_scaled, anom_idx=pkl['y_pred_idx'],
              anomaly_label=anomaly_label, 
              title="{} feature with a total of {} anomalies".format(
                  clf_name, len(pkl['y_pred_idx'])))
    # Run on test set
    if np.isnan(df_filtered_scaled_testing.any()).any():
        cmn.error('df_filtered_scaled_testing.isna > 0 in ' + clf_name)
        df_filtered_scaled_testing.fillna(0)
    if not np.isfinite(df_filtered_scaled_testing.any()).any():
        cmn.error('df_filtered_scaled_testing.isna > 0 in ' + clf_name)
        with pd.option_context('mode.use_inf_as_null', True):
            df_filtered_scaled_testing = df_filtered_scaled_testing.dropna()
    print("Anomalies in test dataset of size {} found: {}".format(
        len(df_filtered_scaled_testing),
        len(pkl['y_pred_idx_test'])))
    l_plot(df=df_filtered_scaled_testing, anom_idx=pkl['y_pred_idx_test'], 
           anomaly_label=anomaly_label,
           title="{} feature with a total of {} anomalies, test data".format(
               clf_name, len(pkl['y_pred_idx_test'])))
    #
    decision_boundary(pkl['clf'],
                      df_filtered_scaled,
                      anomaly_label=1)
    print("="*15)
    return pkl

def plot_anom_vs_features(df, title, anom=None, method='kurtosis', fname=None):
    if 'feature' in df.columns:
        fig = df[['feature', 'total', 'fail interval']].plot(x='feature')
    else:
        fig = df[['total', 'fail interval']].plot()
    fig.set_ylabel('Number of anomalies')
    fig.set_xlabel('Number of features used for training')
    plt.title("Feature selection method: {}".format(method), fontsize=14)
    # plt.legend(fontsize=14)
    plt.rc('legend',fontsize=15) # using a size in points
    plt.suptitle(title, y=1.0, fontsize=18)
    if anom:
        plt.axhline(y=anom, color = 'r', linestyle = ':')
    if fname:
        plt.savefig(fname, format='pdf',
                    bbox_inches='tight', pad_inches=0)
