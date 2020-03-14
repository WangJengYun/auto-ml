import lightgbm as lgb
from hyperopt import fmin,tpe,partial
from auto_ML.setting_for_optimizer import get_param_space,param_transform
from skopt import gp_minimize

# 優化器
class optimizer(object):
    
    def __init__(self,model_type,optimizer_setting,cv_score_func):
        
        self.model_type = model_type
        self.cv_score_func = cv_score_func
        self.optimizer_setting = optimizer_setting
        self.optimizer_type = self.optimizer_setting['optimizer_type']
        self.n_iter = self.optimizer_setting['n_iter']
        self.param_space = get_param_space(self.model_type,self.optimizer_type)
    # 根據參數空間進行搜尋最佳參數值   
    def search_best_param(self):
        best_param = None
        if self.optimizer_type == 'hyperopt':

            algo  = partial(tpe.suggest, n_startup_jobs=-1) 
            optim_result = fmin(self.object_func,self.param_space,\
                                    algo=algo, max_evals=self.n_iter)
            best_param = param_transform(optim_result,self.model_type)
            #print(best_param)
        elif self.optimizer_type == 'skopt':
            self.param_name = [param.name for param in self.param_space]
            optim_process = gp_minimize(self.object_func, self.param_space,
                                        n_calls=self.n_iter,verbose = True,
                                        n_jobs = -1)
            best_param = dict(zip(self.param_name,optim_process.x))
        return(best_param)                       
    # 設定不同的框架的objective function或者說loss function 
    def object_func(self,param): 
        if self.optimizer_type == 'hyperopt':          
            input_param = param_transform(param,self.model_type)
            #print(input_param)
        elif self.optimizer_type == 'skopt':
            input_param = dict(zip(self.param_name,param))
        
        best_score = self.cv_score_func(input_param)
        return best_score

