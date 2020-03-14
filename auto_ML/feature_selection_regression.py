import copy
import pandas as pd 
import numpy as np 
from auto_ML.utils import print_information
from sklearn.base import BaseEstimator
from auto_ML.base import TransformerMixin
from sklearn.feature_selection import GenericUnivariateSelect,RFECV,SelectFromModel
from auto_ML.config import feature_selection_config_dict
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.feature_selection import GenericUnivariateSelect,RFECV,SelectFromModel
from sklearn.feature_selection  import f_regression,mutual_info_regression

# 取得不同的衡量指標
def get_score_func(score_func_name):
    score_func = None
    if score_func_name == 'mutual_info':
        score_func = mutual_info_regression
    elif score_func_name == 'f_regression':
        score_func = f_regression
    else :
        raise ValueError("estimator must be in ('f_regression','mutual_info') but is %s"%(score_func_name))
    return(score_func)

# 取得不同的估計模型
def get_estimator(estimator_name):

    estimator = None
    estimator_param = feature_selection_config_dict["estimator_param"]
    
    if estimator_name == "RandomForestRegressor":
        estimator = RandomForestRegressor(**estimator_param['RandomForestRegressor'])
    elif estimator_name == "ExtraTreesRegressor":
        estimator = RandomForestRegressor(**estimator_param['ExtraTreesRegressor'])
    else:
        raise ValueError("estimator must be in ('RandomForestRegressor','ExtraTreesRegressor') but is %s"%(estimator_name))
    
    return estimator

# 在Embedded中，定義訓練模型
class SelectFromModel(BaseEstimator):

    def __init__(self,estimator_name,threshold_by_number):
        self.estimator_name = estimator_name
        self.threshold_by_number = threshold_by_number
    
    def fit(self,X,y = None):
        self.model = get_estimator(self.estimator_name)    
        self.model.fit(X,y)
        
        self.feature_importances = self.model.feature_importances_
        self.num_cols = len(self.feature_importances)

        return self
    # 取得所選取的重要前幾個特徵
    def get_support(self):
        selected_idx = np.argsort(self.feature_importances)[::-1][:self.threshold_by_number]

        selected_bool = np.repeat(False,self.num_cols)
        selected_bool[selected_idx] = True
        
        return(selected_bool)

# 特徵選取的物件       
class FeatureSelectionTransformer(BaseEstimator,TransformerMixin):
    
    def __init__(self,method = 'KeepAll',estimator = None):
        self.method = method
        self.estimator = estimator
    # 根據設定，進行特徵選取
    def fit(self,X,y = None):
        print_information('Starting Feature Selection')        
        
        self.colnames = X.columns.tolist()
        self.selector = FeatureSelectionTransformer.\
            get_feature_selection_model(self.method,self.estimator)

        if self.selector == 'KeepAll':
            num_cols = X.shape[1]
            self.support_mask = [True for col in range(num_cols)]
        else: 
            self.selector.fit(X,y)
            self.support_mask = self.selector.get_support()
        
        self.index_mask = [idx for idx , not_mask in enumerate(self.support_mask) if not_mask == True]

        print_information('Ending Feature Selection')        
        return self 
    
    # 基於選取的特徵，將資料進行轉換
    def transform(self,X,y = None):
        return X.iloc[:,self.index_mask] 
    # 根據設定進行選取型，進行訓練
    @classmethod
    def get_feature_selection_model(cls,model_name,estimator_name = None):
        feature_selection_model = None
        model_param = copy.deepcopy(feature_selection_config_dict['model_param'])

        if model_name == 'Embedded':
            feature_selection_model = \
                SelectFromModel(estimator_name,
                                **model_param['Embedded'])
        elif model_name == 'Wrapper':
            feature_selection_model = \
                RFECV(estimator = get_estimator(estimator_name),
                        **model_param['Wrapper'])
        elif model_name == 'Filter':
            model_param['Filter']['score_func'] = get_score_func(model_param['Filter']['score_func'])
            feature_selection_model = GenericUnivariateSelect(**model_param['Filter'])
        elif model_name == 'KeepAll':
            feature_selection_model = 'KeepAll'
        else:
            raise ValueError("estimator must be in ('Embedded','Wrapper','Filter','KeepAll') but is %s"%(model_name))

        return feature_selection_model
    # 取的重要變數重要性(目前只有Embedded方法)
    def feature_importances(self):
        importance_table = None
        if self.method == 'Embedded':
            importance_value = self.selector.feature_importances
            importance_table = pd.DataFrame({'colnames':self.colnames ,
                                             'important_value':np.round(importance_value,6)}).\
                                sort_values(by='important_value',ascending=False)
        
        return importance_table


if __name__ == "__main__":
    import pickle 
    with open('./data/Advantech.pickle','rb') as file3:
        Advantech_df = pickle.load(file3)

    y  = Advantech_df['train_train_set'][0]
    X  = Advantech_df['train_train_set'][1].drop(columns = 'REQUIRED_DATE_YM')

    result = FeatureSelectionTransformer('Filter').fit(X,y)
    
   
    result_1= FeatureSelectionTransformer.get_feature_selection_model('Filter').fit(X,y)
   