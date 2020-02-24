from scipy.stats.stats import pearsonr
from sklearn.neural_network import MLPRegressor
import numpy as np
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class MLP_Regressor():
    
    def __init__(self, activation='relu', regularization = 0.005, batch_size=128,
             hidden_layer_sizes=(4096, 2048, 1024,512, 256, 128), learning_rate='adaptive',
             learning_rate_init=0.001, max_iter=25, n_iter_no_change=10,
             optimizer='adam', early_stopping=True, tol=0.0001, validation_fraction=0.15):
        
        self.model = MLPRegressor(activation=activation, alpha=regularization, 
                                  batch_size=batch_size,
                                  hidden_layer_sizes=hidden_layer_sizes, 
                                  learning_rate=learning_rate,
                                  learning_rate_init=learning_rate_init, 
                                  max_iter=max_iter,
                                  n_iter_no_change=n_iter_no_change, solver=optimizer, 
                                  early_stopping=early_stopping,
                                  tol=tol, validation_fraction=validation_fraction)
        
    def fit(self, x, y):
        
        self.model.fit(x, y)
    
    def predict(self, x_test, y_test=None):
        
        y_pred = self.model.predict(x_test)
        
        if y_test != None:
            return y_pred
        else:
            pearson, rmse_score = self.get_scores(y_test, y_pred)
            return y_pred, pearson, rmse_score
            
    
    def get_scores(self,  y_test, y_pred):
        pearson = pearsonr(y_test, y_pred)[0]
        rmse_score = rmse(y_pred, y_test)
        
        return pearson, rmse_score
        
        
        