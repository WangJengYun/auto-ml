import pandas as pd 
import numpy as np 
import lightgbm as lgb 
import xgboost as xgb
from sklearn.base import BaseEstimator
from auto_ML.config import get_model_config
from auto_ML.optimizer import optimizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from abc import ABC,abstractmethod

# 基本模型設定，一些操作設定
class base_model(BaseEstimator,ABC):
    
    def __init__(self,model_type):
        self.model_type = model_type
        self.setting_dict = get_model_config(self.model_type)
        self.cv_setting = self.setting_dict['cv_setting']
        self.optimizer_setting = self.setting_dict['optimizer_setting']
        self.base_param = self.setting_dict['base_param']
        self.fixed_param = self.setting_dict['fixed_param']
    
        if 'orther_param' in self.setting_dict.keys():
            self.orther_param = self.setting_dict['orther_param']
            self.metric_name = None
            if self.orther_param['metric_name']:
                self.metric_name = self.orther_param['metric_name']
            else:
                self.metric_name = self.base_param['metric'][0]
                    
        #if 'fit_param' in self.setting_dict.keys():
        #    self.fit_param = self.setting_dict['fit_param']
        #    self.verbose_eval = 500
        #    if self.fixed_param:
        #        if 'num_trees' in self.fixed_param:
        #            self.fit_param['early_stopping_rounds'] = None
        #            self.verbose_eval = int(self.fixed_param['num_trees']/10)
    def fit(self,train_dataset,valid_dataset = None,cate_feats = None):
        if 'fit_param' in self.setting_dict.keys():
            self.fit_param = self.setting_dict['fit_param']
            self.verbose_eval = 500
            if self.fixed_param:
                if 'num_trees' in self.fixed_param:
                    self.fit_param['early_stopping_rounds'] = None
                    self.verbose_eval = int(self.fixed_param['num_trees']/10)

    @abstractmethod
    def training_model(self,param):
        pass
    
    def find_best_param(self):
        if self.fixed_param:
            best_param = self.fixed_param
        else:
            best_param = optimizer(self.model_type,self.optimizer_setting,self.cv_score).\
                search_best_param()
        return(best_param)
    
    @abstractmethod
    def cv_score(self,param):        
        pass
    
    @abstractmethod
    def predict(self,X):
        pass

# 根據sklearn框架及功能，設計訓練模型及交叉驗證
class sklearn_model(base_model):
    
    def fit(self,train_dataset):
        
        self.X = train_dataset[0]
        self.y = train_dataset[1]
        self.best_param = self.find_best_param()
        #print(self.best_param)
        self.model = self.training_model(self.best_param).fit(self.X,self.y)
        return self

    def cv_score(self,param):        
        cv_model  = self.training_model(param)
        cv_result = cross_val_score(cv_model,self.X,self.y,
                                    cv = self.cv_setting['nfold'],
                                    scoring = self.fit_param['scoring'],
                                    n_jobs = -1)
        #print(cv_result)        
        return np.mean(np.array(cv_result))    
    
    def predict(self,X):
        result = self.model.predict(X)
        return result

# 根據lightgbm的框架，而設定訓練模型及交叉驗證
class lgb_model(base_model):  

    def __init__(self):
        model_type = 'lightgbm'
        super().__init__(model_type)

    def fit(self,train_dataset,valid_dataset = None,cate_feats = 'auto'):
        super().fit(train_dataset,valid_dataset,cate_feats)
        self.valid_sets = []
        self.lgbtrain = lgb.Dataset(data = train_dataset[0].values,label=train_dataset[1],
                    feature_name=train_dataset[0].columns.tolist(),categorical_feature = cate_feats,
                    free_raw_data = False)
        self.valid_sets.append(self.lgbtrain)
        
        if valid_dataset:
            self.lgbvalid = lgb.Dataset(data = valid_dataset[0],label=valid_dataset[1],
                feature_name = valid_dataset[0].columns.tolist(),categorical_feature = cate_feats,
                free_raw_data = False)
            self.valid_sets.append(self.lgbvalid)

        best_param = self.find_best_param()
        self.best_param,self.model = self.training_model(best_param)  
        return self 

    def training_model(self,param):
        
        param.update(self.base_param)
         
        model = lgb.train(param,self.lgbtrain,
                    valid_sets = self.valid_sets,
                    **self.fit_param,verbose_eval = self.verbose_eval)
        if self.fit_param['early_stopping_rounds']:
            param.update({'num_trees':model.best_iteration})
        
        return param,model

    def cv_score(self,param):
        param.update(self.base_param)
        cv_result = lgb.cv(param,
                            train_set = self.lgbtrain,
                            stratified = False if self.base_param['objective'] == 'regression' else True,
                            nfold  = self.cv_setting['nfold'],
                            **self.fit_param)
        best_score = cv_result[self.metric_name+'-mean'][-1]
        return(best_score)
    
    def predict(self,X):
        result = self.model.predict(X)
        return result

# 根據xgb的框架，而設定訓練模型及交叉驗證
class xgb_model(base_model):
    
    def __init__(self):
        model_type = 'xgb'
        super().__init__(model_type)
    
    def fit(self,train_dataset,valid_dataset = None,cate_feats = 'auto'):
        super().fit(train_dataset,valid_dataset,cate_feats)
        self.valid_sets = []
        self.xgbtrain = xgb.DMatrix(data = train_dataset[0].values,label=train_dataset[1],
                    feature_names=train_dataset[0].columns.tolist(),nthread = -1)
        self.valid_sets.append((self.xgbtrain,'train'))

        if valid_dataset:
            self.xgbvalid = xgb.DMatrix(data = valid_dataset[0].values,label=valid_dataset[1],
                    feature_names=valid_dataset[0].columns.tolist(),nthread = -1)
            self.valid_sets.append((self.xgbvalid,'valid'))

        best_param = self.find_best_param()
        self.best_param,self.model = self.training_model(best_param)  
        return self 

    def training_model(self,param):
        param.update(self.base_param)
       
        if "num_trees" in param.keys():
            self.fit_param['num_boost_round'] = param['num_trees']
            if self.fit_param['early_stopping_rounds'] != None:
                self.fit_param['early_stopping_rounds'] = None

        model = xgb.train(param,self.xgbtrain,
                    evals  = self.valid_sets,
                    **self.fit_param,verbose_eval = self.verbose_eval)

        if self.fit_param['early_stopping_rounds']:
            param.update({'num_trees':model.best_iteration})
        
        return param,model

    def cv_score(self,param):
        param.update(self.base_param)
        cv_result = xgb.cv(param,
                            dtrain = self.xgbtrain,
                            stratified = False if self.base_param['objective'] == 'reg:squarederror' else True,
                            nfold  = self.cv_setting['nfold'],
                            **self.fit_param)                            
        best_score = cv_result['test-'+self.metric_name+'-mean'].tail(1).values[0]
        return(best_score)
    
    def predict(self,X):
        self.xgbpred = xgb.DMatrix(data = X.values,
                    feature_names= X.columns.tolist(),nthread = -1)
        result = self.model.predict(self.xgbpred)
        return result

# 使用sklearn框架，建立隨機森林的模型
class rf_model(sklearn_model):
    
    def __init__(self):
        model_type = 'randomforest'
        super().__init__(model_type)
 
    def training_model(self,param):
        model = RandomForestRegressor(**param,n_jobs = -1)
        return model

# 使用sklearn框架，建立決策的模型
class DT_model(sklearn_model):
    
    def __init__(self):
        model_type = 'DT'
        super().__init__(model_type)
 
    def training_model(self,param):
        model = DecisionTreeRegressor(**param)
        return model

# 使用sklearn框架，建立彈性網絡的模型
class ElasticNet_model(sklearn_model):
    
    def __init__(self):
        model_type = 'elasticnet'
        super().__init__(model_type)
 
    def training_model(self,param):
        model = ElasticNet(**param)
        return model

if __name__ == "__main__":
    
    import pickle 
    with open('./data/Advantech.pickle','rb') as file3:
         Advantech_df = pickle.load(file3)
    
    train_X = Advantech_df['train_train_set'][1].drop(columns = 'REQUIRED_DATE_YM')
    train_y = Advantech_df['train_train_set'][0]
    
    valid_X = Advantech_df['train_valid_set'][1].drop(columns = 'REQUIRED_DATE_YM')
    valid_y = Advantech_df['train_valid_set'][0]
    
    #AA = lgb_model().fit((train_X,train_y),None)
    #AA = lgb_model().fit((train_X,train_y),(valid_X,valid_y))
    #AA.best_param
    #BB = xgb_model().fit((train_X,train_y),(valid_X,valid_y))
    #BB.best_param
    #CC = rf_model().fit((train_X,train_y))
    #CC
    #DD = ElasticNet_model().fit((train_X,train_y))
    #DD[0]
    # EE = DT_model().fit((train_X,train_y))
    # EE
