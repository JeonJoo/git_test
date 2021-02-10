# Module Name : pima_indian_diabetes.py

# 데이터 변수명 참고
#Pregnancies: 임신 횟수
# Glucose: 포도당 부하 검사 수치
# BloodPressure: 혈압(mm Hg)
# SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
# Insulin: 혈청 인슐린(mu U/ml)
# BMI: 체질량지수 (체중(kg) / 키(m)^2)
# DiabetesPedigreeFunction: 당뇨 내력 가중치 값
# Age: 나이 > 구간화하자
# Outcome: 클래스 결정 값 (0 또는 1)

# 분석
import pandas as pd
import numpy as np
# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
#머신러닝
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, classification_report,precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Binarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

import warnings
warnings.filterwarnings(action = "ignore")

df = pd.read_csv("diabetes.csv")
print(df.shape)
print(df.info())
print(df.head())
print(df["Outcome"].value_counts())    # 정상:당뇨 = 2:1 (약)
# 결측치가 없다, 전체 데이터 수치형 >>> 가공할 필요가 없음

#df.hist()
#plt.show()
# 이상치 존재의심 :  Glucose 포도당 부하 검사 수치, BloodPressure 혈압(mm Hg), BMI
## 나이 구간화 필요
### 데이터가 편중 되어 있음(정규화 - 스케일링, 아웃라이어)
#### 타켓 : Outcome (0 or 1)

X = df.drop("Outcome", axis = 1)
y = df["Outcome"]
# print(X, y) 잘 나눠짐


# 결측, object없으면 -> 분석시작 가능
# oh.ENcoding()  pd.getDummy() :결측 + 인코딩(글자->수치)

# 상관분석
#plt.figure(figsize=(10,10))
#sns.heatmap(data=df.corr(), annot = True, fmt = ".2g", cmap = "Blues")
#plt.show()
# Pregnancies, Glucose, BMI, Age 주요 항목
## 유전적인 요인은 상관도가 적다고 볼 수 있다.

def get_score(y_test, pred, str = None):
    print("------{}-------".format(str))
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)

    print("정확도 {:.4f}  f1 {:.4f}  정밀도 {:.4f}  재현율 {:.4f}  roc_auc {:.4f}".format(acc, f1, precision, recall, roc_auc))
    # 재현율이 가장 중요 FN > 정상으로 예측 실제론 당뇨
    cf_matrix = confusion_matrix(y_test, pred)
    print(cf_matrix)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 121)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)
get_score(y_test, pred, "최초점수")



# 전처리 Data Preprocessing / 피쳐가공(Feature Engineering)
## Scaler(정규화, 스케일링), outlier(이상치), 피쳐 병합 및 삭제, 구간화(범주화)

# 0인 열을 확인
for col in X.columns:
    cnt = X[col][X[col]==0].count()
    print(col, cnt, np.round(cnt/X.shape[0]*100, 2))

# print(X[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].describe())

zero_col = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_col_median = X[zero_col].median().round(1)
X[zero_col] = X[zero_col].replace(0, zero_col_median)

rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)
get_score(y_test, pred, "0 처리 후 점수")
# print(X[zero_col].describe()) 이상치 전후확인
# for col in X.columns:
#     cnt = X[col][X[col]==0].count()
#     print(col, cnt, np.round(cnt/X.shape[0]*100, 2))    # 0 값이 제대로 수정 되었음을 확인 하였음

## 임신 > 0 나올 수 있음,, 포도당, 혈압, 삼두지방두께, 인슐린, BMI 는 이상치 처리 해야함.
### 삭제, 채우기(일괄->평균,중위,최빈), 예측

# 나이 구간화, 원핫인코딩
X["Age_cate"] = X["Age"].apply(lambda x : int(x//10))
# print(X[["Age_cate", "Age"]])
X = pd.get_dummies(data = X , columns = ["Age_cate"], prefix = "OH_Age_cate")    # OH_Age_cate1 : 10대 -> 없는이유: 데이터에 가장 어린 사람은 27살이다.
#print(X.info())
#print(X.head())

rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)
get_score(y_test, pred, "나이 구간화,원핫인코딩 후 점수")


## boxplot 이상여부 >> drop
# 스케일링/ 정규화 : RobustScaler, MinMaxScaler, StandardScaler -> 가급적 이상치 제거 : 그래야 효과가 있음   >>> 데이터 양이 너무 적으므로 그냥 진행
                    # 중위값 기준
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size = 0.2, random_state = 121)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)
get_score(y_test, pred, "스케일링 된 점수")    # 돌릴 때 마다 들쑥 날쑥 >>> 여러번 돌리자 >>> k-fold


### precision_recall_curv : 임계치 확인
proba = rf_model.predict_proba(X_test)    # 0 또는 1이 나올 각각의 확률, 같을때는 0
precision, recall, th = precision_recall_curve(y_test, proba[:, 1])
print(len(precision), len(recall), len(th))    # 개수 차이가 있으므로 개수를 맞춘후에 차트를 그려보자

plt.plot(th, precision[:len(th)], label = "precision")
plt.plot(th, recall[:len(th)], label = "recall")
plt.xlabel("threadshold")
plt.ylabel("precision & recall value")
plt.legend()
plt.grid()
plt.show()    # 임계값 >>교차지점


## roc_curve : FPR/ TPR
fpr, tpr, th = roc_curve(y_test, proba[:,1])
plt.plot(fpr, tpr, label = "ROC")
plt.plot([0,1], [0,1], label = "th:0.5")
plt.title(auc)
plt.xlabel("FPR")
plt.ylabel("TPR(recall)")
plt.grid()
plt.show()

## 임계치 튜닝을 통한 점수 보정
my_th = [.4, .43, .45, .47, .49, .51, .53]
for th in my_th:
    print("N : P", th, 1-th)
    rf_model.fit(X_train, y_train)
    pred = rf_model.predict(X_test)
    proba = rf_model.predict_proba(X_test)
    get_score(y_test, pred)

    bn = Binarizer(threshold = th)    # 임계치를 기준으로 >>> 여기서는 0보다 크면 1 작으면 0이다.
    fit_trans = bn.fit_transform(proba[:,1].reshape(-1,1))
    auc = roc_auc_score(y_test, proba[:,1].reshape(-1,1))
    print(auc)

# confusion matrix 를 이용하여 검증함.

# 데이터 적은 경우 >> 데이터 증강 K-Fold, st.K-Fold, cross_val_score, GridSearchCV
## 데이터 증강 후 튜닝 GridSearchCV

# my_score={"acc":"accuracy", "f1":"f1"}
#
# my_hyper_param = {  "n_estimators"      :[100] ,
#                     "max_depth"        :[7,9,11],
#                     "min_samples_leaf" :[3,5,7,9],
#                     "random_state" : [121,]
#                  }
#
# gcv_model = GridSearchCV(rf_model, param_grid=my_hyper_param, scoring=my_score, refit="f1", cv=5, verbose=0)
#---- 이하 학습 동일 --------------------
# fit : 학습하다
# gcv_model.fit(X_train, y_train)
# # predict : 시험
# print("best_estimator_", gcv_model.best_estimator_)
# print("best_params_",    gcv_model.best_params_)
# print("best_score_" ,    gcv_model.best_score_)
#
# print("GridSearchCV 평균 정확도 : " , gcv_model.cv_results_["mean_test_acc"].mean())  #mean_test_(본인의score키값)
# print("GridSearchCV 평균 F1 : "    , gcv_model.cv_results_["mean_test_f1"].mean())
#
# gcv_df = pd.DataFrame(gcv_model.cv_results_)
# print(gcv_df.info())
# print("GridSearchCV 평균 정확도 : " , gcv_df["mean_test_acc"].mean())
# print("GridSearchCV 평균 F1 : "    , gcv_df["mean_test_f1"].mean())

### 결과 ... 개쓰레기  갖다버려야함 ㅋㅋㅋㅋ >>> 위에서는 오버피팅





