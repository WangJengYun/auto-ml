from auto_ML.evaluation_metrics import customizing_score

# 各個模型的參數設定，包含項目如下:
# base_setting : 交叉驗證的設定與模型優化的設定
# base_param : 模型的基本設定，主要這會是你訓練的任務而固定，不會放入優化的參數範圍
# fit_param : 訓練過程中的設定，也是事先決定，不會放入優化的參數範圍
# fixed_param : 為固定參數，如果事先設定則不會進行優化，反之為None會自動進行優化

feature_selection_config_dict = {
    'estimator_param' : {
        'RandomForestRegressor':{
            "max_depth":10,
            "n_estimators":15,
            "n_jobs":-1
        },
        'ExtraTreesRegressor':{
            'n_estimators': 100,
            "n_jobs":-1
        }        
    },
    'model_param' : {
        'Embedded':{
            'threshold_by_number':10,
            },
        'Wrapper':{
            "step":0.1,
            "n_jobs":-1
        },
        'Filter':{
            'score_func':'mutual_info',
            'mode':'k_best',
            'param':50
        },
        'KeepAll':None
    }}

def get_model_config(model_type):
    setting_dict = dict()
    base_setting = {
        'cv_setting':{
            'nfold':3
        },
        'optimizer_setting':{
            'optimizer_type':'hyperopt',
            #'optimizer_type':'skopt',
            'n_iter':3
        }
    }
    setting_dict.update(base_setting)
    # setting  of model
    model_setting = None
    if model_type == 'lightgbm':
        metric_name,metric_func= customizing_score(model_type,'WMAPE')
        model_setting = {
            'orther_param':{'metric_name':metric_name},
            'base_param':{'boosting': 'gbdt',
                            'objective': 'regression', 
                            'metric': ['mape'],
                            'n_jobs': -1},
            'fit_param':{"num_boost_round":5000,
                        'early_stopping_rounds':30,
                        'feval':metric_func,
                        #'callbacks':[lgb.reset_parameter(learning_rate=\
                        #    partial(learning_rate_decay,base_lr = .1,power_value = .9995))],
                        #"verbose_eval" : 10},
                        },
            'fixed_param':{
                            'bagging_fraction': 0.43310240726832455,
                            'feature_fraction': 0.3548379046859415,
                            'lambda_l2': 0.675970059229584,
                            'max_depth': 7,
                            'num_trees': 141,
                            'num_leaves': 236,
                            }
            #'fixed_param':None
        }
    elif model_type == "xgb":
        metric_name,metric_func= customizing_score(model_type,'WMAPE')
        model_setting = {
            'orther_param':{'metric_name':metric_name},
            'base_param':{
                'booster':'gblinear',
                'objective':'reg:squarederror',
                'eval_metric':'rmse',
                'nthreads':-1
            },
            'fit_param':{
                'num_boost_round':5000,
                'early_stopping_rounds':30,
                'feval':metric_func,
            },
            "fixed_param":{
               'max_depth':5,
               'num_trees': 191
            }
            #'fixed_param':None
        }
    elif model_type == 'randomforest':
        metric_name,metric_func= customizing_score(model_type,'WMAPE')
        model_setting = {
            "base_param":{
                'task':'Regressor',
                'n_jobs':-1
            },
            'fit_param':{
                'scoring':metric_func
            },
            "fixed_param":{
                'n_estimators':131,
                'max_depth':6,
            }
            #'fixed_param':None
        }
    elif model_type == 'DT':
        metric_name,metric_func= customizing_score(model_type,'WMAPE')
        model_setting = {
            "base_param":{
                'task':'Regressor',
                'n_jobs':-1
            },
            'fit_param':{
                'scoring':metric_func
            },
            #"fixed_param":{
            #    'max_features' :'sqrt',
            #}
            'fixed_param':None
        }
    elif model_type == 'elasticnet':
        metric_name,metric_func= customizing_score(model_type,'WMAPE')
        model_setting = {
            "base_param":{
                'n_jobs':-1
            },
            'fit_param':{
                'scoring':metric_func
            },
            #"fixed_param":{
            #    'l1_ratio' : 0.5
            #}
            'fixed_param':None
        }

        
    setting_dict.update(model_setting)
    return(setting_dict)

