from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1):
        self.param = param

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X * self.param


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('custom_transformer', CustomTransformer(param=2)),
    ('classifier', LogisticRegression())
])
