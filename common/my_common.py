#Module Name : my_common.py

# 1. EDA  : 데이터 사전 탐색
#   0) Dataframe 살펴보기
#   1) 차트그리기
# 2. 결측치/이상치 --> 처리
# 3. 피쳐 전처리/가공
# 4. 학습/평가 : f1, auc
# 5. 검증(GridSearchCV, confusion_matrix)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score,precision_score,recall_score, roc_curve, classification_report,precision_recall_curve
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler, Binarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

import warnings
warnings.filterwarnings(action="ignore")

#------------------- Dataframe 확인
def CHECK_DATAFRAME(df, target="target") :
    print(df.shape)
    print(df.info())
    print(df.head())
    print(df[target].value_counts())

    y = df[target]
    X = df.drop(target, axis=1)
    print(X.shape, y.shape)
    return X, y
# X,y = CHECK_DATAFRAME(df)



# -------------------------------------
# 4-2. IQR : 25%~75% 범위 값
# -------------------------------------
def CHECK_OUTLIER(df=None, column=None):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    Q1 = np.percentile(df[column].values, 25)
    Q3 = np.percentile(df[column].values, 75)
    IQR = Q3 - Q1
    IQR_weight = IQR * 1.5
    minimum = Q1 - IQR_weight
    maximum = Q3 + IQR_weight
    outlier_idx = df[column][  (df[column]<minimum) | (df[column]>maximum)  ].index
    return outlier_idx
# oulier_idx = CHECK_OUTLIER(df=df, column=col)
# print(col, oulier_idx)
#df.drop(outlier_idx, axis=0, inplace=True)
#---------------------------------------------
# numeric_columns = df.dtypes[df.dtypes != 'object'].index
# for i, col in enumerate(numeric_columns) :
#     oulier_idx = CHECK_OUTLIER(df=df, column=col)
#     print(col , oulier_idx)
#     #df.drop(outlier_idx, axis=0, inplace=True)


def CHECK_ZERO(X):
    for col in X.columns:
        gcnt = X[col][X[col] == 0].count()
        print(col, gcnt, np.round(gcnt / X.shape[0] * 100, 2))


def CHECK_NULL(X):
    # print(df.isnull().sum())
    for col in X.columns:
        gcnt = X[col][X[col] == np.nan].count()
        print(col, gcnt, np.round(gcnt / X.shape[0] * 100, 2))


def SCORES(y_test, pred, proba, str=None) :
    print("------{}-------".format(str))
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    # rocauc = roc_auc_score(y_test, pred)
    auc = roc_auc_score(y_test, proba[:, 1].reshape(-1, 1))
    print("정확도{:.4f}  F1 {:.4f}=(정밀도{:.4f}  재현률{:.4f}) auc{:.4f}".format(acc, f1, precision, recall, auc))
    cf_matrix = confusion_matrix(y_test, pred)
    print(cf_matrix)


def SPLIT_FIT_PREDICT(model, X, y, str=None, rate=0.2) :
    X_train ,X_test , y_train, y_test = train_test_split(X, y, test_size=rate, random_state=121)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    SCORES(y_test, pred, proba, str)
# rf_model = RandomForestClassifier()
# SPLIT_FIT_PREDICT(rf_model, X, y, str="최초점수", 0.5)
# SPLIT_FIT_PREDICT(rf_model, X, y)


def CHART_HEATMAP(df,targetClass, topn=0):
    plt.figure(figsize=(10,6))
    if topn>0 :
        idx = df.corr().nlargest(topn, targetClass).index
        sns.heatmap(df[idx].corr(), annot=True,  fmt=".2g", cmap="Blues")
        print(idx)
    elif topn<0 :
        idx = df.corr().nsmallest(-1*topn, targetClass).index
        sns.heatmap(df[idx].corr(), annot=True, fmt=".2g", cmap="Blues")
        print(idx)
    else :
        sns.heatmap(df.corr(), annot=True, fmt=".2g", cmap="Blues")
    plt.show()

# CHART_HEATMAP(df, 5)
# CHART_HEATMAP(df, -5)
# CHART_HEATMAP(df)


def CROSS_VALIDATION(model, X, y, cv=5) :
    my_score={"acc":"accuracy", "f1":"f1"}
    score_list = cross_validate(model, X, y, scoring=my_score, cv=5, verbose=0)
    score_df = pd.DataFrame(score_list)
    print("cross_validation 평균 정확도 : " , score_df["test_acc"].mean())
    print("cross_validation 평균 f1 : " , score_df["test_f1"].mean())
# CROSS_VALIDATION(model, X, y)

def GRIDSEARCHCV(my_hyper_param, model, X_train, y_train) :
    my_score = {"acc": "accuracy", "f1": "f1"}
    gcv_model = GridSearchCV(rf_model, param_grid=my_hyper_param, scoring=my_score, refit="f1", cv=5, verbose=0)
    #---- 이하 학습 동일 --------------------
    # fit : 학습하다
    gcv_model.fit(X_train, y_train)
    print("best_estimator_", gcv_model.best_estimator_)
    print("best_params_",    gcv_model.best_params_)
    print("best_score_" ,    gcv_model.best_score_)
    print("GridSearchCV 평균 정확도 : " , gcv_model.cv_results_["mean_test_acc"].mean())  #mean_test_(본인의score키값)
    print("GridSearchCV 평균 F1 : "    , gcv_model.cv_results_["mean_test_f1"].mean())
# my_hyper_param = {  "n_estimators"     :[100]}
# GRIDSEARCHCV(my_hyper_param, model, X_train, y_train)


def GROUP_FILLNA_MEAN(bycol, cols, tcol):
    dict = df.groupby(by=[bycol])[cols].mean().astype(np.int32).to_dict()
    fill_mean_func = lambda gname: gname.fillna(dict[bycol][gname[tcol]])
    df = df.groupby(bycol).apply(fill_mean_func)
    return df
# df["Name2"] = df["Name"].str.extract("([A-Za-z]+)\.")
# dict = df.groupby(by=["Name2"])[["Name2","Age"]].mean().astype(np.int32).to_dict()
# fill_mean_func = lambda gname: gname.fillna(dict['Age'][gname.name])
# df = df.groupby('Name2').apply(fill_mean_func)
# df = GROUP_FILLNA_MEAN("Name2",["Name2","Age"],"name"


def REPLACE(target_df, a, b) :
    target_df = target_df.replace(a, b)
    return target_df
#DF_REPLACE(X[["Insulin", "BMI"]], 0, X[["Insulin", "BMI"]].median())
#X["BMI"] = DF_REPLACE(X["BMI"], 0, 99)

def SCALER(scaler_model):
    scaler  = scaler_model
    X_scaler = scaler.fit_transform(X)
    return X_scaler
# X_scaler = SCALER(StandardScaler())

def CHART_PRECISION_RECALL_CURV(y_test, proba):
    precision, recall, th = precision_recall_curve(y_test, proba[:, 1])
    print(len(precision), len(recall), len(th))
    plt.plot(th, precision[:len(th)], label="precision")
    plt.plot(th, recall[:len(th)], label="recall")
    plt.xlabel("threadshold")
    plt.ylabel("precision & recall value")
    plt.legend()  # plt.legend(["precision","recall"])
    plt.grid()
    plt.show()


def CHART_ROC_CURV(y_test, proba):
    fpr, tpr, th = roc_curve(y_test, proba[:, 1])
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], label='th:0.5')
    plt.title(auc)
    plt.xlabel("FPR")
    plt.ylabel("TPR(recall)")
    plt.grid()
    plt.show()
