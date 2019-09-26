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