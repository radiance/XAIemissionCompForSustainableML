# This is a sample Python script.
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn import preprocessing

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

"""
"    Title: Energy Efficiency Dataset
"    Author: UJJWAL CHOWDHURY
"    Date: 14.6.2022
"    Code version: 5.0
"    Availability: https://www.kaggle.com/code/ujjwalchowdhury/energy-efficiency-dataset/notebook
"
"""
def importing_and_exploring_data():
    data = pd.read_csv('energy_efficiency_data.csv')
    data.info()

    # Preview correlation
    plt.figure(figsize=(12, 12))
    sns.heatmap(data.corr(), annot=True)

    num_list = list(data.columns)

    fig = plt.figure(figsize=(10, 30))

    for i in range(len(num_list)):
        plt.subplot(15, 2, i + 1)
        plt.title(num_list[i])
        plt.hist(data[num_list[i]], color='blue', alpha=0.5)

    plt.tight_layout()

    sns.pairplot(data)

    sns.distplot(data['Cooling_Load'], hist=False)
    sns.distplot(data['Heating_Load'], hist=False)
    plt.legend(['Cooling Load', 'Heating Load'])
    plt.xlabel('Load')
    plt.savefig('cooling_heating_load.png')
    return data

"""
*    Title: Energy Efficiency Dataset
*    Author: UJJWAL CHOWDHURY
*    Date: 14.6.2022
*    Code version: 5.0
*    Availability: https://www.kaggle.com/code/ujjwalchowdhury/energy-efficiency-dataset/notebook
*
"""
def data_preprocessing(data):
    df = data.copy()
    X = df[['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
            'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution']]
    y_h = df[['Heating_Load']]
    y_c = df[['Cooling_Load']]

    # define standard scaler
    mmx_scaler = MinMaxScaler()
    stand_scalar = StandardScaler()
    # transform data
    X_normalized = mmx_scaler.fit_transform(X)
    X_standarized = stand_scalar.fit_transform(X)
    return X_normalized, y_h, y_c

"""
*    Title: Energy Efficiency Dataset
*    Author: UJJWAL CHOWDHURY
*    Date: 14.6.2022
*    Code version: 5.0
*    Availability: https://www.kaggle.com/code/ujjwalchowdhury/energy-efficiency-dataset/notebook
*
"""
def model_fitting(X_normalized, y_h, y_c, do_feature_reduction=False):
    X_train, X_test, yh_train, yh_test, yc_train, yc_test = train_test_split(X_normalized, y_h, y_c,
                                                                             test_size=0.33, random_state=42)
    Acc = pd.DataFrame(index=None,
                       columns=['model', 'train_Heating', 'test_Heating', 'train_Cooling', 'test_Cooling'])

    regressors = [['SVR', SVR()],  # waterfall
                  ['DecisionTreeRegressor', DecisionTreeRegressor()],
                  ['KNeighborsRegressor', KNeighborsRegressor()],
                  ['RandomForestRegressor', RandomForestRegressor()],  # waterfall
                  ['MLPRegressor', MLPRegressor()],  # waterfall
                  ['AdaBoostRegressor', AdaBoostRegressor()],  # waterfall
                  ['GradientBoostingRegressor', GradientBoostingRegressor()],  # waterfall
                  ['LinearRegression', LinearRegression()]
                  ]
    if do_feature_reduction:
        return feature_reduction(regressors, X_train, X_test, yh_train, yh_test, yc_train, yc_test)
    else:
        for mod in regressors:
            name = mod[0]
            model = mod[1]

            model.fit(X_train, yh_train)
            actr1 = r2_score(yh_train, model.predict(X_train))
            acte1 = r2_score(yh_test, model.predict(X_test))

            model.fit(X_train, yc_train)
            actr2 = r2_score(yc_train, model.predict(X_train))
            acte2 = r2_score(yc_test, model.predict(X_test))
            
            #print('=============================\n', name)
            #print('MSE heating train: %.3f, test: %.3f' % (mean_squared_error(yh_train, yh_train_pred), mean_squared_error(yh_test, yh_test_pred)))
            #print('R^2 heating train: %.3f, test: %.3f' % (r2_score(yh_train, yh_train_pred), r2_score(yh_test, yh_test_pred)))
            #print('MSE cooling train: %.3f, test: %.3f' % (mean_squared_error(yc_train, yc_train_pred), mean_squared_error(yc_test, yc_test_pred)))
            #print('R^2 cooling train: %.3f, test: %.3f' % (r2_score(yc_train, yc_train_pred), r2_score(yc_test, yh_test_pred)))
            #print('r2_score (heating,cooling)')

            Acc = Acc.append(pd.Series(
                {'model': name, 'train_Heating': actr1, 'test_Heating': acte1, 'train_Cooling': actr2,
                 'test_Cooling': acte2}), ignore_index=True)

        Acc.sort_values(by='test_Cooling')
        Acc.to_csv('energy_effiency_regressor_comparison_without_fr.csv')
        return regressors


def shap_analysis_waterfall(model, X):
    X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution
    explainer = shap.Explainer(model.predict, X100)
    shap_values = explainer(X)
    sample_ind = 18
    shap.plots.waterfall(shap_values[sample_ind], max_display=14)


def shap_analysis_beeswarm(model, X):
    X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution
    explainer = shap.Explainer(model.predict, X100)
    shap_values = explainer(X)
    sample_ind = 18
    shap.plots.beeswarm(shap_values[sample_ind], max_display=14)


def shap_analysis_tree(model, X):
    shap.initjs()
    X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    sample_ind = 18
    html = shap.force_plot(explainer.expected_value, shap_values[0, :])
    shap.save_html("force_plot.html", html)


def feature_reduction(regressors, X_train, X_test, yh_train, yh_test, yc_train, yc_test):
    Acc = pd.DataFrame(index=None, columns=['model', 'train_Heating', 'test_Heating',
                                            'train_Cooling', 'test_Cooling', 'num_features'])
    column_names = ["Relative_Compactness", "Saurface_Area", "Wall_Area", "Roof_Area",
                                               "Overall_Height", "Orientation", "Glazing_Area",
                                               "Glazing_Area_Distribution"]

    df_Xtrain = pd.DataFrame(X_train, columns=column_names)
    df_Xtest = pd.DataFrame(X_test, columns=column_names)
    features_ranked_h, features_ranked_c = feature_ranking(df_Xtrain, yh_train, yc_train)

    for mod in regressors:
        name = mod[0]
        model = mod[1]

        df_Xtrain_h = df_Xtrain.copy()
        df_Xtest_h = df_Xtest.copy()
        for index, row in features_ranked_h.iterrows():
            df_Xtrain_h.drop(labels=index, axis=1, inplace=True)
            df_Xtest_h.drop(labels=index, axis=1, inplace=True)

            model.fit(df_Xtrain_h, yh_train)
            actr1 = r2_score(yh_train, model.predict(df_Xtrain_h))
            acte1 = r2_score(yh_test, model.predict(df_Xtest_h))

            num_features = len(df_Xtrain_h.columns)
            Acc = Acc.append(pd.Series(
                {'model': name, 'train_Heating': actr1, 'test_Heating': acte1, 'train_Cooling': 0,
                 'test_Cooling': 0, 'num_features': num_features}), ignore_index=True)
            if num_features <= 1:
                break

        df_Xtrain_c = df_Xtrain.copy()
        df_Xtest_c = df_Xtest.copy()
        for index, row in features_ranked_c.iterrows():
            df_Xtrain_c.drop(labels=index, axis=1, inplace=True)
            df_Xtest_c.drop(labels=index, axis=1, inplace=True)

            model.fit(df_Xtrain_c, yc_train)
            actr2 = r2_score(yc_train, model.predict(df_Xtrain_c))
            acte2 = r2_score(yc_test, model.predict(df_Xtest_c))

            num_features = len(df_Xtrain_c.columns)
            Acc = Acc.append(pd.Series(
                {'model': name, 'train_Heating': 0, 'test_Heating': 0, 'train_Cooling': actr2,
                 'test_Cooling': acte2, 'num_features': num_features}), ignore_index=True)
            if num_features <= 1:
                break

    Acc.sort_values(by='test_Cooling')
    Acc.to_csv('energy_effiency_regressor_comparison.csv')
    return regressors


def feature_ranking(X, y_h, y_c):
    lab_enc = preprocessing.LabelEncoder()
    encoded_h = lab_enc.fit_transform(y_h)
    encoded_c = lab_enc.fit_transform(y_c)

    model = RandomForestClassifier()
    rfe_h = RFE(model, n_features_to_select=5)
    rfe_h.fit(X, encoded_h)
    rfe_c = RFE(model, n_features_to_select=5)
    rfe_c.fit(X, encoded_c)

    ranking_h = pd.DataFrame(rfe_h.ranking_, index=X.columns, columns=["Rank"])
    ranking_h.sort_values(by="Rank", inplace=True, ascending=False)

    ranking_c = pd.DataFrame(rfe_c.ranking_, index=X.columns, columns=["Rank"])
    ranking_c.sort_values(by="Rank", inplace=True, ascending=False)

    return ranking_h, ranking_c


def main(do_shap=True, do_feature_reduction=False):
    data = importing_and_exploring_data()
    X_normalized, y_h, y_c = data_preprocessing(data)
    mod = model_fitting(X_normalized, y_h, y_c, do_feature_reduction)
    if (do_shap):
        shap_analysis_tree(mod[1][1], X_normalized)
    # shap_analysis_beeswarm(mod[1][1], X_normalized)
# waterfall
# shap_analysis_waterfall(mod[0][1], X_normalized)
# shap_analysis_waterfall(mod[3][1], X_normalized)
# shap_analysis_waterfall(mod[4][1], X_normalized)
# shap_analysis_waterfall(mod[5][1], X_normalized)
# shap_analysis_waterfall(mod[6][1], X_normalized)
# for model in mod:
#     shap_analysis_waterfall(model[1], X_normalized)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
