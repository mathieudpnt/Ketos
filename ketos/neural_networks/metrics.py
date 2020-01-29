from sklearn.metrics import  accuracy_score, recall_score, precision_score, average_precision_score, fbeta_score
import numpy as np

def precision_recall_accuracy_f(y_true, y_pred, f_beta=1.0):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    epsilon = 0.000001
    
    a = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    f = (1.0 +f_beta**2)*p*r / ((f_beta**2*p)+r+epsilon)
    #import pdb;pdb.set_trace()
    precision=p
    recall=r
    f_score=f
    accuracy=a

    return {"f_score":f_score, "precision":precision,"recall":recall, "accuracy":accuracy}



class FScore():
    def __init__(self, onehot=True, beta=1.0):
        self.onehot = onehot
        self.beta = beta
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

        epsilon = 0.000001
    
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)

        f_score = (1.0 + self.beta**2)*p*r / ((self.beta**2*p)+r+epsilon)
        
        return f_score 



class Accuracy():
    def __init__(self, onehot=True):
        self.func = accuracy_score
        self.onehot = onehot
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return self.func(y_true, y_pred)
      


class Precision():
    def __init__(self, onehot=True):
        self.func = precision_score
        self.onehot = onehot
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return self.func(y_true, y_pred)
       


class Recall():
    def __init__(self, onehot=True):
        self.func = recall_score
        self.onehot = onehot
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return self.func(y_true, y_pred)




