from auto_ML.modeling import *
from auto_ML.evaluation_metrics import WMAPE
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import KFold
import numpy as np 
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# ensemble的參數設定
config = {
    "ensemble_type":'stacking',
    "CV":{
        'num_cv':3
    },
    'blending':{
        "model_list":[lgb_model(),xgb_model(),rf_model()],
        "top_k":3
    },
    'stacking':{
        "model_list":[lgb_model(),xgb_model()],
        'meta_model':lgb_model(),
        "use_features_in_secondary":False,
        'split_ratio': 0.33 
    }}

# 單一個模型的交叉驗證
class customized_CV(object):
    def __init__(self,empty_model,num_cv,dataset):
        
        self._dataset = dataset
        self._num_cv = num_cv
        self._empty_model = empty_model
        len_index = dataset[0].shape[0]
        index_list = []
        for train_idx,valid_idx in KFold(n_splits=self._num_cv,shuffle = True).split(range(len_index)):
            index_list.append((train_idx,valid_idx))
        self._index_list = index_list
        self.predict_value = np.zeros(len_index)
        self._score = []
        self.cv_model = []
    # 執行交叉驗證
    def run(self,parallel = False):

        if parallel:
            pass
        else:
            for i in range(self._num_cv):
                valid_index,cv_valid_pred,score,cv_model = self.run_single_subset(i)
                self._score.append(score)
                self.cv_model.append(cv_model)
                self.predict_value[valid_index] = cv_valid_pred
        return(self)
    
    # 使用單一個資料集來訓練模型
    def run_single_subset(self,i):
        X,y = self._dataset
        train_index,valid_index = self._index_list[i]
        #print(train_index,valid_index)
        cv_train_X = X.iloc[train_index,:]
        cv_valid_X = X.iloc[valid_index,:]
        cv_train_y = y[train_index]
        cv_valid_y = y[valid_index]
        
        print('Start to train model for CV %d'%i)
        cv_model = self._empty_model.fit((cv_train_X,cv_train_y))
        cv_valid_pred = cv_model.predict(cv_valid_X)
        score = WMAPE(cv_valid_pred,cv_valid_y)
        print(score)
        
        return valid_index,cv_valid_pred,score,cv_model
    
    # 取得分數、(OOF)預測值及交叉驗證模型
    def get_result(self):
        return np.mean(self._score), self.predict_value,self.cv_model

# 集成學習
class ensemble_learning(object):
    
    def __init__(self):
        self.ensemble_type = config['ensemble_type']
        self.CV =  config['CV']
        
        if self.ensemble_type == 'blending':
            self.model_list = config['blending']['model_list']
            self.top_k = config['blending']['top_k']
        elif  self.ensemble_type == 'stacking':
            self.model_list = config['stacking']['model_list']
            self.meta_model = config['stacking']['meta_model']
            self.use_features_in_secondary = config['stacking']['use_features_in_secondary']
            self.split_ratio =  config['stacking']['split_ratio']
            self.out_of_Fold_pred = []

        self.model_result = []
    # 根據設定執行blending與stacking方法
    def fit(self,dataset):
        self.dataset = dataset
        X,y = self.dataset
        if self.ensemble_type == 'blending':
            score,select_model_B = self._select_top_K()
            print(score,select_model_B)
            
            selec_model_list = list(np.array(self.model_list)[select_model_B])
            for model in selec_model_list:
                training_model = model.fit(dataset)
                self.model_result.append(training_model)
        
        elif self.ensemble_type == 'stacking':
            for model in self.model_list:
                CV_model = customized_CV(model,self.CV['num_cv'],self.dataset).run()
                score,valid_pred,cv_model = CV_model.get_result()
                self.model_result.append(cv_model)
                self.out_of_Fold_pred.append(valid_pred)
            
            self.meta_features = np.column_stack(self.out_of_Fold_pred)

            if self.use_features_in_secondary:
                self.meta_features = np.hstack([X,self.meta_features])
            
            self.meta_X = pd.DataFrame(self.meta_features)
            self.meta_X.columns = ['F_'+str(i) for i in self.meta_X.columns]
            # fit meta model
            meta_model_S_empty = clone(self.meta_model)
            meta_model_S_empty.fixed_param = None
            if self.split_ratio == 0 :
                self.meta_model_F = meta_model_S_empty.fit((self.meta_X,y))
            else:
                X_train, X_test, y_train, y_test = train_test_split(self.meta_X,y,test_size = self.split_ratio)
                best_param = meta_model_S_empty.fit((X_train,y_train),(X_test,y_test)).best_param
                meta_model_F_empty = clone(self.meta_model)
                meta_model_F_empty.fixed_param = best_param
                self.meta_model_F = meta_model_F_empty.fit((X_train,y_train))

            self.Final_pred = self.meta_model_F.predict( self.meta_X)

        return self 
    # 這個主要是blending的方法，選取前幾個表現較好的模型   
    def _select_top_K(self):
        all_model_score = []
        for model in self.model_list:
            CV_model = customized_CV(model,self.CV['num_cv'],self.dataset).run()
            score,_,_ = CV_model.get_result()
            all_model_score.append(score)
        
        all_model_rank_score = rankdata(all_model_score)
        select_model = (all_model_rank_score <=  self.top_k)
        return all_model_score,select_model
    
    # 根據blenfing與stacking架構進行預測
    def predict(self,X):
        predict_value = None
        
        if self.ensemble_type == 'blending':
            predict_features = np.zeros(shape = (X.shape[0],len(self.model_result)))
            for idx,model in enumerate(self.model_result):
                pred_value = model.predict(X)
                predict_features[:,idx] = pred_value
            predict_value = predict_features.mean(axis = 1)
        
        elif self.ensemble_type == 'stacking':
            predict_features = np.zeros(shape = (X.shape[0],len(self.model_result)))
            for idx,model_set in enumerate(self.model_result):
                single_model_predict = np.column_stack([ model.predict(X) \
                                                        for model in model_set])
                predict_features[:,idx] = single_model_predict.mean(axis = 1)
            
            if self.use_features_in_secondary:
                predict_features = np.hstack([X,predict_features])
                predict_features = pd.DataFrame(predict_features)
                predict_features.columns = ['F_'+str(i) for i in predict_features.columns]
            
            predict_value = self.meta_model_F.predict(predict_features)
        
        return predict_features,predict_value

if __name__ == "__main__":
    
    import pickle 
    with open('./data/Advantech.pickle','rb') as file3:
         Advantech_df = pickle.load(file3)
    
    train_X = Advantech_df['train_train_set'][1].drop(columns = 'REQUIRED_DATE_YM')
    train_y = Advantech_df['train_train_set'][0]
    
    valid_X = Advantech_df['train_valid_set'][1].drop(columns = 'REQUIRED_DATE_YM')
    valid_y = Advantech_df['train_valid_set'][0]
    
    A = ensemble_learning().fit((train_X,train_y))
    predict_table = A.predict(valid_X)
    B = predict_table[1]
    B[B<0] = 0
    print(WMAPE(B,valid_y))