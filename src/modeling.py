import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,confusion_matrix

def make_models():
    return {
        "logistic_regression": Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("model",LogisticRegression(max_iter=3000,solver="liblinear",random_state=42))
        ])
    }

def make_logistic_pipeline(C=1.0, class_weight="balanced"):
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()),
        ("model",LogisticRegression(C=C,max_iter=3000,solver="liblinear",class_weight=class_weight,random_state=42))
    ])

def evaluate(y,p,threshold=0.5):
    pred=np.asarray(p)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p),"accuracy":accuracy_score(y,pred),
    "precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),
    "f1":f1_score(y,pred,zero_division=0),"balanced_accuracy":balanced_accuracy_score(y,pred),
    "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
