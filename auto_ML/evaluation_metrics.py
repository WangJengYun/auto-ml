import numpy as np 
from sklearn.metrics import make_scorer

# WMAPE的計算公式
def WMAPE(preds, target,is_log = False):
    if is_log:
        preds = np.expm1(preds)
        target = np.expm1(target)
    MAPE = np.abs(target - preds)/(target + 1)
    MAPE[MAPE >= 1] = 1
    term1 = np.sum(MAPE*(target + 1))
    result = (term1/(np.sum(target)+1))*100
    if (result<0)|(result>=100):
        result = 100
    #print(result)
    return(result)
# lightgbm框架的WMAPE
def lgb_wmape_score(preds,data,is_log = False):
                labels = data.get_label()
                WMAPE_val = WMAPE(preds,labels,is_log)
                return('WMAPE',WMAPE_val,False)
# XGB框架的WMAPE
def xgb_wmape_score(preds,data,is_log = False):
        labels = data.get_label()
        WMAPE_val = WMAPE(preds,labels,is_log)
        return [('WMAPE',WMAPE_val)]
# 選取客製化的衡量指標
def customizing_score(model_type,metric):
    if metric == 'WMAPE':
        if model_type == 'lightgbm':
            metric_func = lgb_wmape_score
        elif model_type == 'xgb':
            metric_func = xgb_wmape_score
        else:
            metric_func = make_scorer(WMAPE)
    return (metric,metric_func)

def learning_rate_decay(current_iter,base_lr,power_value):
    lr =  base_lr*np.power(power_value,current_iter)
    return lr if lr > 1e-3 else 1e-3
 