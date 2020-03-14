# 訓練模型且轉換
class TransformerMixin:

    def fit_transform(self, X, y=None, **fit_params):
      
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X)

