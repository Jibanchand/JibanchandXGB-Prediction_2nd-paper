import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import os
import pickle
from models import Adaboost, XGBoost, Bagging, RandomForest
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
import xgboost as xgb


dataset_name = '5.xlsx'
import pandas as pd

import pandas as pd

file_path = os.path.join(r"C:\Users\J\Desktop\Manuscript\PHD  code and results\Settlement\datasets\hypertuning", str(dataset_name))
data = pd.read_excel(file_path)


#labels = {"B":"Footing Width, B", "D": "Depth", "L/B" : "Footing Geometry, L/B", "Y": "Unit weight of soil, Y", "Theta": "Angle of internal friction", "Qu":"Ultimate Bearing Capacity, Qu"}

def Preprocess(random_state,do_scale, basePath):
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X,y,indices , test_size = 0.2, random_state=random_state)
    find_stats(X,y,X_train,y_train,X_test,y_test, random_state, basePath)
    if do_scale:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    return (X,y,X_train, X_test, y_train, y_test, id_train, id_test)

def save_indices(id_train,id_test,basePath):
    new = []
    new.append(id_train)
    new.append(id_test)
    with open(basePath+"/indices.txt", "wb") as fp:   #Pickling
        pickle.dump(new, fp)

def Train_Model(X_train, y_train, X_test, y_test, model_name, hypertuning):
    if hypertuning:
        if model_name == 'XGBRegressor':
            model, best_hyperparams = XGBoost.train_model(X_train, y_train, X_test, y_test)
        elif model_name == 'BaggingRegressor':
            model, best_hyperparams = Bagging.train_model(X_train, y_train, X_test, y_test)
        elif model_name == 'AdaBoostRegressor':
            model, best_hyperparams = Adaboost.train_model(X_train, y_train, X_test, y_test)
        elif model_name == 'RandomForestRegressor':
            best_hyperparams = {'criterion': 'squared_error'}  # Update the criterion to a valid value
            model = RandomForestRegressor(criterion=best_hyperparams['criterion'])
            model.fit(X_train, y_train)
        return (model, best_hyperparams)
    elif hypertuning == False:
        if model_name == 'XGBRegressor':
            model = xgb.XGBRegressor(random_state=42)
            model.fit(X_train, y_train)
        elif model_name == 'BaggingRegressor':
            model = BaggingRegressor()
            model.fit(X_train, y_train)
        elif model_name == 'AdaBoostRegressor':
            model = AdaBoostRegressor()
            model.fit(X_train, y_train)
        elif model_name == 'RandomForestRegressor':
            model = RandomForestRegressor(criterion='squared_error')  # Update the criterion to a valid value
            model.fit(X_train, y_train)
        return model

def find_stats(X,y,X_train,y_train,X_test,y_test, random_state, basePath):
    agg = dict()
    EDA_list = {}
    for i in range(len(X[0])+1):
        agg_dict = dict()
        if i != int(len(X[0])):
            agg_dict['All_max'] = np.max(X[:,i])
            agg_dict['All_min'] = np.min(X[:,i])
            agg_dict['All_mean'] = np.mean(X[:,i])
            agg_dict['All_std'] = np.std(X[:,i])
            agg_dict['Train_max'] = np.max(X_train[:,i])
            agg_dict['Train_min'] = np.min(X_train[:,i])
            agg_dict['Train_mean'] = np.mean(X_train[:,i])
            agg_dict['Train_std'] = np.std(X_train[:,i])
            agg_dict['Test_max'] = np.max(X_test[:,i])
            agg_dict['Test_min'] = np.min(X_test[:,i])
            agg_dict['Test_mean'] = np.mean(X_test[:,i])
            agg_dict['Test_std'] = np.std(X_test[:,i])
            agg_dict['Train_cov'] = str(round(np.cov(X_train[:,i],y_train)[0][1],2))
            agg_dict['Test_cov'] = str(round(np.cov(X_test[:,i],y_test)[0][1],2))
            agg_dict['Train_corr'] = str(round(np.corrcoef(X_train[:,i],y_train)[0][1],2))
            agg_dict['Test_corr'] = str(round(np.corrcoef(X_test[:,i],y_test)[0][1],2))
            
        if i == int(len(X[0])):
            agg_dict['All_max'] = np.max(y)
            agg_dict['All_min'] = np.min(y)
            agg_dict['All_mean'] = np.mean(y)
            agg_dict['All_std'] = np.std(y)
            agg_dict['Train_max'] = np.max(y_train)
            agg_dict['Train_min'] = np.min(y_train)
            agg_dict['Train_mean'] = np.mean(y_train)
            agg_dict['Train_std'] = np.std(y_train)
            agg_dict['Test_max'] = np.max(y_test)
            agg_dict['Test_min'] = np.min(y_test)
            agg_dict['Test_mean'] = np.mean(y_test)
            agg_dict['Test_std'] = np.std(y_test)
        agg[data.columns[i]] = agg_dict
    EDA_list[random_state] = agg
    with open(basePath +'/Aggregate.txt','w') as fp:
        fp.writelines(json.dumps(EDA_list)+'\n')

def plot_corr_scatter(X_train, X_test, y_train, y_test, basePath):
    plt.figure(figsize = (15,15))
    #plt.suptitle("Train and test set distribution of input vs output params", size = 20)
    plt.tight_layout()
    for i in range(X_train.shape[1]):
        plt.subplot(3,3,i+1)
        plt.xlabel(data.columns[i], fontdict={'fontsize':14})
        plt.ylabel(str(data.columns[-1]), fontdict={'fontsize':14})
        plt.scatter(X_train[:,i],y_train, label = 'train')
        plt.scatter(X_test[:,i],y_test, label = 'test')
        plt.legend()
    plt.savefig(basePath+'/Split Params vs S.png', bbox_inches = 'tight', dpi= 300)
    plt.close()

def plot_dist_scatter(X_train, y_train, X_test, y_test , basePath):
    plt.figure(figsize = (15,15))
    #plt.suptitle("Train and test set distribution of input vs output params", size = 20)
    for i in range(X_train.shape[1]):
        plt.subplot(3,3,i+1)
        plt.ylabel(data.columns[i], fontdict={'fontsize': 14})
        plt.xlabel('Index', fontdict={'fontsize': 14})
        plt.scatter(np.arange(0,X_train.shape[0]),X_train[:,i], label = 'train')
        plt.scatter(np.arange(0,X_test.shape[0]),X_test[:,i], label = 'test')
    plt.subplot(3,3,i+2)
    plt.ylabel('S', fontdict={'fontsize': 14})
    plt.xlabel('Index', fontdict={'fontsize': 14})
    plt.scatter(np.arange(0,y_train.shape[0]),y_train, label = 'train')
    plt.scatter(np.arange(0,y_test.shape[0]),y_test, label = 'test')
    plt.legend()
    plt.savefig(basePath+'/Split Distro.png', bbox_inches = 'tight', dpi= 300)
    plt.close()  


def residual_plot(y_true,y_pred, basePath):
    res = (y_true - y_pred)
    plt.figure(figsize = (8,8))
    plt.xlabel('Fitted values', fontdict = {'fontsize': 14})
    plt.ylabel('Residuals', fontdict = {'fontsize': 14})
    plt.scatter(y_pred, res)
    plt.savefig(basePath+'/Residual Plot.png', bbox_inches = 'tight', dpi = 300)
    plt.close()

def actual_predicted_plot(y_train, y_true,y_pred_train,y_pred,r2,r2_train, basePath, model_name):
    a,b = np.polyfit(y_true, y_pred, 1)
    c,d = np.polyfit(y_train,y_pred_train,1)
    textX = int(max(np.max(y_true),np.max(y_train)))
    textY = int(min(np.min(y_true),np.min(y_train)))
    plt.figure(figsize = (8,8))
    plt.xlabel('Actual values', fontdict = {'fontsize': 14})
    plt.ylabel('Predicted values', fontdict = {'fontsize': 14})
    #plt.axline([0, 0], [1, 1], color = 'gray', label = 'actual = predicted')
    plt.scatter(y_train, y_pred_train, label = 'Training set', marker = '^')
    plt.scatter(y_true, y_pred, label = 'Test set', marker = 'o')
    plt.plot(y_train, c*y_train + d, linestyle = ':', label = 'best fit train', color = 'purple')
    plt.plot(y_true, a*y_true + b, label = 'best fit test', color = 'green')
    if model_name == 'AdaBoostRegressor':
        plt.text(textX-10, textY+8, f'Training R\N{SUPERSCRIPT TWO} score = {r2_train:.3f}', ha='right', va='top')
        plt.text(textX-11.3, textY+5, f'Testing R\N{SUPERSCRIPT TWO} score = {r2:.3f}', ha='right', va='top')
    else:    
        plt.text(textX-10, textY+5, f'Training R\N{SUPERSCRIPT TWO} score = {r2_train:.3f}', ha='right', va='top')
        plt.text(textX-11.3, textY+2, f'Testing R\N{SUPERSCRIPT TWO} score = {r2:.3f}', ha='right', va='top')
    plt.legend()
    plt.savefig(basePath+'/Actual Predicted Plot.png', bbox_inches = 'tight', dpi = 300)
    plt.close()
'''
Code to save the predicted and actual sampels side by side. 
'''
def save_prediction(X,y, model, basePath, test = False):
    y_pred = model.predict(X)
    temp = pd.DataFrame(np.concatenate([y.reshape(y.shape[0],1), y_pred.reshape(y.shape[0],1), X.reshape(X.shape[0],int(len(data.columns)-1))], axis = 1), columns=list(['True', 'Predicted']+list(data.columns[:-1])))
    if test: 
        temp.to_excel(basePath+'/Test_prediction.xlsx')
    else: 
        temp.to_excel(basePath+'/Train_prediction.xlsx')



def plot_feature_importance(importance, basePath):
    '''
    This function plots the feature importance of the trained models. 
    '''
    features = list(data.columns[:-1])
    fig = plt.figure(figsize = (7,5.5))
    plt.yticks(range(len(features)),features)
    plt.xticks(np.arange(0, max(importance)+.25, 0.1))
    plt.xlabel('Feature Importance', fontdict={"fontsize":16})
    plt.margins(x= .2)
    bars = plt.barh(features,importance)
    for bar in bars:
        xval = bar.get_width()
        x_height = xval + 0.005
        plt.text(x_height, bar.get_y()+.25, round(xval,3), fontdict={"fontsize":14})
    plt.savefig(basePath+'/feature_importance.png', dpi = 300, bbox_inches = 'tight')
    plt.close()

#-------------------------------------------------------------------------------------------------------------------------------
def find_metrics(X_train,X_test, y_train,y_test, best_hyperparams, model, model_name, basePath):
    basePath = basePath+'/'+model_name
    os.mkdir(basePath)
    model_metrics = dict()
    metrics = dict()
    y_pred = model.predict(X_test)
    y_pred_train=model.predict(X_train)
    metrics['r2'] = r2_score(y_test, y_pred)
    metrics['MAE'] = mean_absolute_error(y_test,y_pred)
    metrics['MSE'] = mean_squared_error(y_test, y_pred)
    metrics['RMSE'] = (mean_squared_error(y_test, y_pred))**0.5
    metrics['MAPE'] = mean_absolute_percentage_error(y_test,y_pred)
    metrics['r'] = round(np.corrcoef(y_test,y_pred)[0][1],2)
    y_train_pred = model.predict(X_train)
    metrics['Train r2'] = r2_score(y_train, y_train_pred)
    metrics['Train MAE'] = mean_absolute_error(y_train,y_train_pred)
    metrics['Train MSE'] = mean_squared_error(y_train, y_train_pred)
    metrics['Train RMSE'] = (mean_squared_error(y_train, y_train_pred))**0.5
    metrics['Train MAPE'] = mean_absolute_percentage_error(y_train,y_train_pred)
    metrics['Train r'] = round(np.corrcoef(y_train,y_train_pred)[0][1],2)
    residual_plot(y_test,y_pred, basePath)
    actual_predicted_plot(y_train,y_test,y_pred_train,y_pred, metrics['r2'] ,metrics['Train r2'],basePath, model_name)
    model_metrics[model_name] = metrics
    filename = basePath+'/'+'model.sav'
    pickle.dump(model, open(filename, 'wb')) #save model weights
    #save model predictions for each sample
    save_prediction(X_train, y_train, model, basePath)
    save_prediction(X_test, y_test, model, basePath, test=True)
    #plot feature importance curve
    if model_name == 'BaggingRegressor':
        importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
        #print(importance.shape)
        plot_feature_importance(importance, basePath)

    else:
        plot_feature_importance(list(model.feature_importances_), basePath)
    # save model metrics
    if best_hyperparams == None: 
        pass

    if best_hyperparams is not None:
        best_hyperparams = {k: v.item() if isinstance(v, np.int64) else v for k, v in best_hyperparams.items()}
        with open(basePath + '/Hyperparams.txt', 'a') as fp:
            fp.writelines(json.dumps(best_hyperparams) + '\n')
        
    with open(basePath+'/Results.txt','a') as fp:
        fp.writelines(json.dumps(model_metrics)+'\n')