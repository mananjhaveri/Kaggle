import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.ensemble import VotingClassifier

def predict(model):
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    print(classification_report(y_test, y_preds))

if __name__ == "__main__":
    
    # import data
    df = pd.read_csv("../data/data.csv")

    # preprocessing
    from preprocessing import preprocess
    pre = preprocess()
    X_train, y_train, X_test, y_test = pre.fit_transform(df, drop_features=True)

    # predict 
    models = [("lr", LogisticRegression()), ("svc", SVC()), ("rfc", RandomForestClassifier()), ('xgb', XGBClassifier())]

    for model in models:
        print(model[0])
        predict(model[1])
        print("\n---------------------------------------\n")

    print("voting clf")
    voting_clf = VotingClassifier(estimators=models, voting="hard")
    predict(voting_clf)