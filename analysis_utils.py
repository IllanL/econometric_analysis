import os
from datetime import datetime
from math import floor, log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import signal
from scipy.stats import norm, pearsonr, rankdata
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, ccf, grangercausalitytests
from statsmodels.tsa import seasonal

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ------------------------------
# Visualization helpers
# ------------------------------

def series_rel_plot(s_1, s_2):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10))
    label1, label2 = ["--" if not x.name else x.name for x in (s_1, s_2)]

    ax1.plot(s_1, color='gold', label=label1)
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(s_2, color='black', label=label2)
    ax2.legend(loc='upper right')

    sns.regplot(x=s_2, y=s_1, color='gold', ax=ax3)

    return s_1.corr(s_2)


def dual_plot_corr(series1, series2):
    fig, ax1 = plt.subplots()
    label1, label2 = ["--" if not x.name else x.name for x in (series1, series2)]

    ax1.plot(series1, 'black', label=label1)
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(series2, 'grey', label=label2)
    ax2.legend(loc='upper right')

    bool_filter = (series1.isna()) | (series2.isna())

    return pearsonr(series1[~bool_filter], series2[~bool_filter])


def ccf_funct(series1, series2, nlags, adjusted=True, both_sides=False):
    arr_length = nlags - 1
    net_setter = 0
    ccf_vals = ccf(series1, series2, adjusted=adjusted, nlags=nlags)

    if both_sides:
        neg_ccf_vals = ccf(series2, series1, adjusted=adjusted, nlags=nlags)[::-1]
        ccf_vals = np.append(neg_ccf_vals[:-1], ccf_vals)
        net_setter = arr_length

    abs_max_pos = abs(ccf_vals).argmax()
    abs_max_pos_net = abs_max_pos - net_setter
    print(abs_max_pos_net)
    print(ccf_vals[abs_max_pos])

    plt.stem(range(-net_setter, nlags), ccf_vals, linefmt='gold', markerfmt='gold', basefmt='grey')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.title('Cross-Correlation:' + series1.name + ' and ' + series2.name)

    plt.axhline(-1.96 / np.sqrt(len(ccf_vals)), color='grey', ls='--')
    plt.axhline(1.96 / np.sqrt(len(ccf_vals)), color='grey', ls='--')
    plt.show()

    return ccf_vals


def lagged_corr_plot(series1, series2, nlags, both_sides=False):
    arr_length = nlags - 1
    net_setter = 0

    corrs_array = np.array([series1.corr(series2.shift(n)) for n in range(nlags)])

    if both_sides:
        neg_corrs_array = np.array([series2.corr(series1.shift(n)) for n in range(nlags)])[::-1]
        corrs_array = np.append(neg_corrs_array[:-1], corrs_array)
        net_setter = arr_length

    plt.stem(range(-net_setter, nlags), corrs_array, linefmt='gold', markerfmt='gold', basefmt='grey')
    plt.xlabel('Lag')
    plt.ylabel('Lagged correlations')
    plt.title('Lagged correlations:' + series1.name + ' and ' + series2.name)

    return corrs_array


def twinplot(plt_dict_1, plt_dict_2, color1='black', color2='grey'):
    fig, ax1 = plt.subplots()

    for val in plt_dict_1.keys():
        ax1.plot(plt_dict_1[val], color=color1, label=val)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    for val in plt_dict_2.keys():
        ax2.plot(plt_dict_2[val], color=color2, label=val)

    ax2.legend(loc='upper right')

    for val1 in plt_dict_1.keys():
        for val2 in plt_dict_2.keys():
            print(pearsonr(plt_dict_1[val1], plt_dict_2[val2]))


def plot_ccf_2(df, col1, col2, limit, interval):
    aux_df = df[[col1, col2]].dropna()[interval[0]:interval[1]]

    x = np.linspace(0, limit - 1, limit)
    y = ccf(aux_df[col1], aux_df[col2], adjusted=False, nlags=limit)
    plt.stem(x, y)

    return y.argmax(), y.max()


def roll_apply_max_ccf(series, df, col1, col2, limit):
    aux_df = df.loc[series.index, [col1, col2]].dropna()
    return ccf(aux_df[col1], aux_df[col2], adjusted=False, nlags=limit).argmax()


def cross_correlate(df, col1, col2, adjusted=True):
    fig, (ax1, plt2) = plt.subplots(2, 1, figsize=(12, 6))

    arr_1 = df[col1].dropna().to_numpy()
    arr_2 = df[col2].dropna().to_numpy()

    arr_1 = (arr_1 - arr_1.mean()) / arr_1.std()
    arr_2 = (arr_2 - arr_2.mean()) / arr_2.std()

    ax1.plot(arr_1, label=df[col1].name)
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(arr_2, '--', label=df[col2].name)
    ax2.legend(loc='upper right')

    cross_corr = signal.correlate(arr_1, arr_2, mode='full')
    lags = signal.correlation_lags(arr_1.size, arr_2.size, mode='full')

    if adjusted:
        adj_array = lags.max() + 1 - abs(lags)
        corr_adj_array = np.array([max(x, 100) for x in adj_array])
        cross_corr = cross_corr / corr_adj_array

    lag = lags[np.argmax(cross_corr)]
    plt2.plot(lags, cross_corr, label='cross-correlation function')
    plt2.plot(lag, cross_corr.max(), 'o')
    plt2.legend()

    return lag


# ------------------------------
# Data preparation helpers
# ------------------------------

def detrender(series):
    coef = (series.dropna().iloc[-1] - series.dropna().iloc[0]) / series.size

    sub_list = coef * np.array(range(0, series.size))
    sub_series = pd.Series(sub_list, index=series.index)

    ret_series = series - sub_series
    ret_series.name = series.name

    return ret_series


def forward_fill_increment(series, step):
    filled = []
    last_val = None
    counter = 0

    for val in series:
        if pd.notna(val):
            last_val = val
            counter = 1
            filled.append(val)
        else:
            if last_val is not None:
                filled_val = last_val + step * counter
                filled.append(filled_val)
                counter += 1
            else:
                filled.append(np.nan)

    return pd.Series(filled, index=series.index)


def append_csv_to_dataframe(path):
    combined_df = pd.DataFrame()
    series_dict = dict()

    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            print(file_name)
            file_path = os.path.join(path, file_name)
            csv_df = pd.read_csv(file_path, index_col=0, parse_dates=True, date_format='%d/%m/%Y')

            series = csv_df.iloc[:, 0]
            series.index = pd.to_datetime(series.index)
            series.name = os.path.splitext(file_name)[0].split(' - ')[1]
            series_dict[series.name] = series.copy()

            mod_series = series.resample(rule='ME').mean().ffill()
            mod_series.index = mod_series.index.to_period('M').to_timestamp()

            combined_df = pd.concat([combined_df, mod_series], axis=1)
            combined_df.sort_index(inplace=True)
    return combined_df, series_dict


# ------------------------------
# Model evaluation helpers
# ------------------------------
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


def bic(n, mse, p):
    return n * np.log(mse) + p * np.log(n)


def evaluate_model(name, model, X_input, y_true):
    y_pred = model.predict(X_input)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, len(y_true), X_input.shape[1])
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    model_bic = bic(len(y_true), mse, X_input.shape[1])

    print(f"\n--- {name} Evaluation ---")
    print(f"MAPE:       {mape:.4f}")
    print(f"Adjusted R²:{adj_r2:.4f}")
    print(f"BIC:        {model_bic:.2f}")
    print(f"RMSE:       {np.sqrt(mse):.4f}")


# ------------------------------
# Regression helpers
# ------------------------------
def lin_model(x, y):
    x = x.dropna()
    y = y.dropna()

    common_index = x.index.intersection(y.index)

    x = x[common_index]
    y = y[common_index]

    x_arr = np.array(x[common_index])
    y_arr = np.array(y[common_index])

    return np.polyfit(x_arr, y_arr, deg=1)


def robust_r2_score(s1, s2):
    common_id = s1.dropna().index.intersection(s2.dropna().index)
    return r2_score(s1[common_id], s2[common_id])


class LinearModelClass:
    def __init__(self):
        self.coefs = None
        self.fixed_y_hat = None
        self.var_y_hat = None
        self.y_hat = None
        self.x = None
        self.y = None
        self.error = None
        self.percentual_error = None
        self.mape = None
        self.max_ape = None

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.coefs = lin_model(x, y)

        self.fixed_y_hat = self.coefs[1]
        self.var_y_hat = self.coefs[0] * self.x
        self.y_hat = self.var_y_hat + self.fixed_y_hat

        self.error = self.y_hat - self.y
        self.percentual_error = self.error / self.y
        self.mape = np.mean(np.abs(self.percentual_error))
        self.max_ape = np.max(np.abs(self.percentual_error))

        self.r2 = robust_r2_score(self.y, self.y_hat)

    def plot_regression(self, scatter_kws=None, line_kws=None):
        common_index = self.x.index.intersection(self.y.index)
        sns.regplot(x=self.y_hat[common_index], y=self.y[common_index], scatter_kws=scatter_kws, line_kws=line_kws)
        plt.xlabel('Predicted (y_hat)')
        plt.ylabel('Actual (y)')
        plt.title('Regression Plot')
        plt.show()

    def plot_prediction(self, color1, color2):
        plt.plot(self.y, color=color1, label='Y')
        plt.plot(self.y_hat, color=color2, label='y_hat', linestyle=':')
        plt.legend()
        plt.title('Series vs Prediction')
        plt.show()

    def plot_perc_error(self, color):
        plt.plot(self.percentual_error, color=color, label='Y', linestyle=':')
        plt.legend()
        plt.title('Percentual error')
        plt.show()

    def plot_prediction_decomposition(self, color1, color2, color3):
        plt.plot(self.y_hat, color=color1)
        plt.plot(self.var_y_hat, color=color2, linestyle='--')
        plt.plot(self.fixed_y_hat, color=color3, linestyle='--')
