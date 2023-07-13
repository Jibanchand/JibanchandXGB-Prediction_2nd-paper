import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from functools import partial


space = { 'max_depth': hp.quniform('max_depth',3,10,1 ),
         'gamma':hp.uniform('gamma',1,9),
         'reg_alpha': hp.quniform('reg_alpha',40,180,1),
         'reg_lambda': hp.uniform('reg_lambda', 0,1),
         'min_child_weight': hp.quniform('min_child_weight',0,10,1),
         'eta': hp.uniform('eta',0.1,0.4)
         }


def objective(space, X_train, y_train, X_test ,y_test):
    #print(type(space))
    clf=xgb.XGBRegressor(max_depth=int(space['max_depth']), 
                        gamma=int(space['gamma']), reg_lambda=int(space['reg_lambda']),
                     reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']), eta=float(space['eta']) )   
    clf.fit(X_train, y_train,verbose=False)
        

    y_pred = clf.predict(X_test)
    score = r2_score(y_test, y_pred)
    #print ("SCORE:", score)
    return {'loss': mean_squared_error(y_test, y_pred), 'status': STATUS_OK ,'Trained_Model': clf}

def train_model(X_train, y_train, X_test, y_test):

    trials = Trials()
    best_hyperparams = fmin(fn = partial(objective, X_train =  X_train, y_train = y_train, X_test = X_test, y_test = y_test),
        space = space,
        max_evals = 150,
        trials = trials,
        algo = tpe.suggest)
    model = trials.results[np.argmin([r['loss'] for r in trials.results])]['Trained_Model']
    return (model, best_hyperparams)