from hyperopt import hp
from skopt.space import Real, Integer ,Categorical
# 設定優化的參數空間，結構如下:
# 參數名稱:[參數類型,參數範圍]
def set_param(model_type):
    param_dict = None
    if model_type == "lightgbm":
        param_dict = {
            'max_depth':['int',(1,15)],
            'num_leaves':['int',(2,20)],
            'bagging_fraction':['float',(0,1)],
            'feature_fraction':['float',(0,1)],
            'lambda_l2':['float',(0,1)]
        }
    elif model_type == "xgb":
        param_dict = {
            "max_depth":['int',(1,15)],
            'colsample_bytree':['float',(0,1)],
            'lambda':['float',(0,1)],
        }
    elif model_type == 'randomforest':
        param_dict = {
            'n_estimators':['int',(1,500)],
            'max_depth':['int',(1,6)]
        }
    elif model_type == 'DT':
        param_dict = {
            'max_features':['str',('auto','sqrt','log2')]
        }
    elif model_type == 'elasticnet':
        param_dict = {
            'l1_ratio':['float',(0,1)]
        }
    return param_dict

# 將參數空間轉換成不同框架的參數空間物件
def get_param_space(model_type,optimazer_type):
    # param_dict = set_param('lightgbm') 
    param_dict = set_param(model_type)
    
    if optimazer_type == 'hyperopt':
        param_space = dict()
        param_names = list(param_dict.keys())
        for param_name in param_names:
            value_type = param_dict[param_name][0]
            value_range = param_dict[param_name][1]
            if value_type == 'int':
                param_space[param_name] = hp.randint(param_name,value_range[1])
            elif value_type == 'float':
                param_space[param_name] = hp.uniform(param_name,value_range[0],value_range[1])
            elif value_type == 'str':
                param_space[param_name] = hp.choice(param_name,list(value_range))
    elif optimazer_type == 'skopt':
        param_space = []
        param_names = list(param_dict.keys())
        for param_name in param_names:
            value_type = param_dict[param_name][0]
            value_range = param_dict[param_name][1]
            if value_type == 'int':
                param_space.append(Integer(value_range[0],value_range[1],name = param_name))
            elif value_type == 'float':
                param_space.append(Real(value_range[0]+10**-6,value_range[1],"log-uniform",name = param_name))
            elif value_type == 'str':
                param_space.append(Categorical(categories = list(value_range),name = param_name))
    return param_space

# 參數空間再進行一層的轉換，避免訓練過程中出錯
def param_transform(single_param_set,model_type):

    if model_type == 'lightgbm':
        single_param_set["max_depth"] = int(single_param_set["max_depth"]) + 1
        single_param_set["learning_rate"] = 0.1
        single_param_set["bagging_fraction"] = single_param_set["bagging_fraction"]+0.0001 
        single_param_set["num_leaves"] = 2 if int(single_param_set["num_leaves"]) <= 1 else int(single_param_set["num_leaves"])
    elif model_type == 'xgb':
        single_param_set["max_depth"] = int(single_param_set["max_depth"]) + 1
    elif model_type == 'randomforest':
        single_param_set["n_estimators"] = int(single_param_set["n_estimators"]) + 1
        single_param_set["max_depth"] = int(single_param_set["max_depth"]) + 1
    elif model_type == 'DT':
        if type(single_param_set["max_features"])!=str:
            single_param_set["max_features"] = ['auto','sqrt','log2'][single_param_set["max_features"]]
    return(single_param_set)
