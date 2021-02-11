import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

df = pd.read_csv("kaggle_titanic_train.csv")
print(df.shape)
print(df.info())
print(df.tail())


#  data, *,
# cmap=None, center=None, robust=False,
#     annot=None, fmt=".2g",
plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(), annot=True, fmt=".2g") #cmap=Blue
# plt.show()


#--------------------------------------
# Target 피쳐 선정
# 1   Survived     891 non-null    int64
#--------------------------------------
X = df.drop("Survived", axis=1)  #문제
y = df["Survived"]               #답안
print(X[:2])
print(y[:2])

#--------------------------------------
# Object 처리
#--------------------------------------
#  3   Name         891 non-null    object  --> Sex로 성별 구분 정도로 활용
#  4   Sex          891 non-null    object  --> lambda 이용해 female:0 , male:1 으로 변경
#  8   Ticket       891 non-null    object  --> 의미있는 데이터로 보기 어렵다고 판단
#  10  Cabin        204 non-null    object  --> 결측이 너무 많아 드롭(687건)
#  11  Embarked     889 non-null    object  -->

X["Sex"] = X["Sex"].apply(lambda x: 0 if x == "female" else 1)
# print(X[["Sex","Sex2"]].head())
print(X["Sex"].head())

#C106 A51 5254 --> 글자1개만 추출  (문법공부)
cp = X[X["Cabin"].isnull() == False].copy()
print(cp["Cabin"].isnull().sum())
X["Cabin2"] =  X["Cabin"].str[0:1]   #.str[0]
print(X["Cabin2"])
print(pd.crosstab(X["Cabin2"], y).T)
print(pd.crosstab(X["Cabin2"], X["Pclass"]).T)
print(pd.crosstab(X["Pclass"], y).T)
print(pd.crosstab([X["Pclass"],X["Sex"]], y).T)

#--------------------------------------
# 결측처리 - 1.삭제   2.대체   3.예측
#--------------------------------------
#   5  Age          714 non-null    float64
#X["Age"].fillna()
# cp = X[X["Age"].isnull() == True].copy()
# cp["Age2"] = cp["Age"].fillna(55)  #.mean()
# print(cp[["Age","Age2"]])

# ----------------------------------------------------------
# 나이를 예측하기 위해 이름의 호칭 추출  SibSp	Parch
# 호칭 별 평균 나이로 Age 결측 데이터 처리
# ----------------------------------------------------------
X["Name2"] = X["Name"].str.extract("([A-Za-z]+)\.")
# fill_mean_func = lambda g: g["Age"].fillna(g.mean())
# X = X.groupby(by=["Name2"]).apply(fill_mean_func)
dict = X.groupby(by=["Name2"])[["Name2","Age"]].mean().astype(np.int32).to_dict()
print(dict['Age'])
print(X[["Name2","Name","Age"]].head(10))
fill_mean_func = lambda gname: gname.fillna(dict['Age'][gname.name])
X = X.groupby('Name2').apply(fill_mean_func)
print(X[["Name2","Name","Age"]].head(10))
# X["Age"] = X["Age"].fillna(30)

# 11 12 13 --> 10 //10  1
# 22 23 24 --> 20 //10  2
# 80       --> 80 //10  8
# 나이 구간화   /  % //
X["Age_cate"] = X["Age"].apply(lambda x : int(x//10))
print(X[["Age_cate", "Age"]])
print(pd.crosstab([X["Pclass"],X["Sex"],X["Age_cate"]], y).T)

#  11  Embarked     889 non-null    object  --> 생존과 무관해보임
print(pd.crosstab(X["Embarked"], y).T)
print(pd.crosstab([X["Embarked"], X["Pclass"]], y).T)

X["Embarked"] = X["Embarked"].apply(lambda x: 1 if x == "C" else (2 if x == "Q" else 3))
print(X["Embarked"])

# 병합 피쳐 : 중복된특징,
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
X["SP"] = X["SibSp"] + X["Parch"]
# print(X[["SP", "SibSp", "Parch"]])


# 삭제 피쳐 : 일련번호
#  0   PassengerId  891 non-null    int64
print(X.shape)
X.drop("PassengerId", axis=1, inplace=True)
print(X.info())

replace_col = ["SibSp", "Parch","Name","Name2","Age"]    #SP=SibSp+Parch     Age_cate<--Name,Name2,Age
del_col = ["Ticket","Cabin","Cabin2","Fare","Embarked"]  #Fare<--Pclass,SP   Embarked
replace_col = replace_col + del_col
X.drop(replace_col, axis=1, inplace=True)
print(X.info())

heat_df = X.copy()
heat_df["Servvvv"] = y
plt.figure(figsize=(10,10))
sns.heatmap(data=heat_df.corr(), annot=True, fmt=".2f") #cmap=Blue
#plt.show()

# -----------------------------------
# 분석 (모델선정/ 평가척도/검증)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121, shuffle=True)

# ??모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()
# xg = XGBoost()

models = [dt_model, knn_model, rf_model]
for model in models :
    # fit : 학습하다
    model.fit(X_train, y_train)
    # predict : 시험
    y_pred = model.predict(X_test)
    # score : 예측 정확도 확인
    score = accuracy_score(y_test, y_pred)
    #-- f1, auc, accuracy, 교차검증
    #-- mse mae rmse
    #-- shilluet
    print(model.__class__)
    print(model.__str__(), ":" , score) #0.9666666666666667

print(X.head())

## 분석(예측)력을 저해하는 원인
# 1. 피쳐가 많은 경우        --> 모델의 복잡도가 증가하는 경우(오버피팅 <-> 언더피팅)
# 2. 수치가 큰 경우         --> log, scalling
# 3. 결측데이터(Null)       --> isnull(), fillna() ,print(X.isnull().sum())
# 4. 이상치(Outlier)       --> 협의 후 삭제/대체
# 5. 데이터가 편중          --> 정규분포화
# 6. 피쳐가공 (Object-->변환,  유니크한 일련번호X, 구간(범주)화,  원핫인코딩)
# 7. 데이터 적은 경우        --> 데이터 증강
# 8. 모델이 적절하지 않는 경우 --> 다른 모델 사용, 튜닝(Hyper Parameter)




# 3. 결측데이터(Null)       --> isnull(), fillna()
print(X.isnull().sum())

# 4. 이상치(Outlier)
# -------------------------------------
# 4-1. box plot , scatter plot
# -------------------------------------
# fig, axes = plt.subplots(nrows=3, ncols=5)
# columns = df.columns  #[....]
# for i, col in enumerate(columns) :
#     r = int(i / 5)
#     c = i % 5
#     sns.boxplot(x=col, y='Survived', data=df, ax=axes[r][c])
# plt.show()

# -------------------------------------
# 4-2. IQR : 25%~75% 범위 값
# -------------------------------------
def get_outlier(df=None, column=None):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    Q1 = np.percentile(df[column].values, 25)
    Q3 = np.percentile(df[column].values, 75)
    IQR = Q3 - Q1
    IQR_weight = IQR * 1.5
    minimum = Q1 - IQR_weight
    maximum = Q3 + IQR_weight
    outlier_idx = df[column][  (df[column]<minimum) | (df[column]>maximum)  ].index
    return outlier_idx

# 함수 사용해서 이상치 값 삭제
numeric_columns = df.dtypes[df.dtypes != 'object'].index
# columns = df.columns  #[....]
for i, col in enumerate(numeric_columns) :
    oulier_idx = get_outlier(df=df, column=col)
    print(col , oulier_idx)
    #df.drop(outlier_idx, axis=0, inplace=True)



# 5. 데이터가 편중          --> 정규분포화
df.hist(figsize=(20,5))
# plt.show()

from sklearn.preprocessing import StandardScaler  # m0 v1
from sklearn.preprocessing import MinMaxScaler    # 0~1
from sklearn.preprocessing import RobustScaler    # min~median~max
scaler = StandardScaler()
#scaler.fit()
#scaler.transform()
X_scaler = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=121, shuffle=True)

models = [dt_model, knn_model, rf_model]
for model in models :
    # fit : 학습하다
    model.fit(X_train, y_train)
    # predict : 시험
    y_pred = model.predict(X_test)
    # score : 예측 정확도 확인
    score = accuracy_score(y_test, y_pred)
    #-- f1, auc, accuracy, 교차검증
    #-- mse mae rmse
    #-- shilluet
    print(model.__class__)
    print(model.__str__(), ":" , score) #0.9666666666666667


# 6. 피쳐가공 (Object-->변환,  유니크한 일련번호X, 구간(범주)화,  원핫인코딩)
# Object-->변환 후 원핫인코딩
print(X["Age_cate"].head())

print(X.info())
X_encoding = pd.get_dummies(data=X , columns=["Age_cate"], prefix = "OH_Age_cate")  #, drop_first = True
print(X_encoding.info())
print(X_encoding.head())
#     (drop_first)
#      Age_cate   OH_0   OH_1  (891, 12)  +9개추가  --> (891,21)
# 0    1           1      0
# 1    1           0      1
X_train, X_test, y_train, y_test = train_test_split(X_encoding, y, test_size=0.2, random_state=121, shuffle=True)
models = [dt_model, knn_model, rf_model]
for model in models :
    # fit : 학습하다
    model.fit(X_train, y_train)
    # predict : 시험
    y_pred = model.predict(X_test)
    # score : 예측 정확도 확인
    score = accuracy_score(y_test, y_pred)
    #-- f1, auc, accuracy, 교차검증
    #-- mse mae rmse
    #-- shilluet
    print(model.__class__)
    print(model.__str__(), ":" , score) #0.9666666666666667



#** 평가 메트릭스            --> f1_score(), roc_auc(), accuracy_score() ,  conf._matrix,
# accuracy_score
# f1_score -- precision  recall
#    (scoring=f1_micro or f1_macro)`
# roc_auc  -- precision  recall --> FPR/TPR



# 7. 데이터 적은 경우        --> 데이터 증강  ==> K-Fold, St.K-Fold, GridSearchCV(증강+튜닝)
# ==> 검증(신뢰) , 대량의 학습으로 예측이 좋아진다




from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True,  random_state=121)
kf = KFold(n_splits=5,random_state=121,shuffle=True)
accuracy_score_list = []
f1_score_list = []
print(X.info())
# for (idx_train, idx_test) in kf.split(X):
# i = 0
for i, (idx_train, idx_test) in enumerate(kf.split(X)):  #skf.split(X, y)

    X_train, X_test = X.iloc[idx_train] , X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train] , y.iloc[idx_test]
    #~~~
    #---- 이하 학습 동일 --------------------
    # fit : 학습하다
    rf_model.fit(X_train, y_train)
    # predict : 시험
    y_pred = rf_model.predict(X_test)
    # score : 예측 정확도 확인
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_score_list.append(accuracy)
    f1       = f1_score(y_test, y_pred)
    f1_score_list.append(f1)
    print(i , ":" , accuracy, f1)
    # i = i+1
print("Kfold 평균 정확도:",np.mean(accuracy_score_list))
print("Kfold 평균 F1:",np.mean(f1_score_list))


# scoring matrix
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
from sklearn.model_selection import cross_val_score
score_list = cross_val_score(rf_model, X, y, scoring="accuracy", cv=5, verbose=0)
print("cross_val_score 평균 F1:",np.mean(score_list))

from sklearn.model_selection import cross_validate
my_score={"acc":"accuracy", "f1":"f1"}
score_list = cross_validate(rf_model, X, y, scoring=my_score, cv=5, verbose=0)
print("score_list------->", score_list)
score_df = pd.DataFrame(score_list)
print(score_df.head(10))
print("cross_validation 평균 정확도 : " , score_df["test_acc"].mean())
print("cross_validation 평균 f1 : " , score_df["test_f1"].mean())




# GridSearchCV(param ) --> 튜닝 전용
# 총 loop 횟수 : 12 = 1* 4* 3
my_hyper_param = {  "n_estimators"     :[100],  #,300] ,
                    "max_depth"        :[3,5,7,9],
                    "min_samples_leaf" :[1,3,5],
                    "random_state"     :[121,]
                 }

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
dd = RandomForestClassifier()
# gcv_model = GridSearchCV(rf_model, param_grid=my_hyper_param, scoring="f1", refit=True, cv=5, verbose=0)
# #---- 이하 학습 동일 --------------------
# # fit : 학습하다
# gcv_model.fit(X_train, y_train)
# # predict : 시험
# print("best_estimator_", gcv_model.best_estimator_)
# print("best_params_",    gcv_model.best_params_)
# print("best_score_" ,    gcv_model.best_score_)

my_score={"acc":"accuracy", "f1":"f1"}
gcv_model = GridSearchCV(rf_model, param_grid=my_hyper_param, scoring=my_score, refit="f1", cv=5, verbose=0)
#---- 이하 학습 동일 --------------------
# fit : 학습하다
gcv_model.fit(X_train, y_train)
# predict : 시험
print("best_estimator_", gcv_model.best_estimator_)
print("best_params_",    gcv_model.best_params_)
print("best_score_" ,    gcv_model.best_score_)

print("GridSearchCV 평균 정확도 : " , gcv_model.cv_results_["mean_test_acc"].mean())  #mean_test_(본인의score키값)
print("GridSearchCV 평균 F1 : "    , gcv_model.cv_results_["mean_test_f1"].mean())


gcv_df = pd.DataFrame(gcv_model.cv_results_)
print(gcv_df.info())
print("GridSearchCV 평균 정확도 : " , gcv_df["mean_test_acc"].mean())
print("GridSearchCV 평균 F1 : "    , gcv_df["mean_test_f1"].mean())





# y_pred = rf_model.predict(X_test)

# confusion-matrix,  auc-roc
from sklearn.metrics import confusion_matrix
confusion_matrix()

# 8. 모델이 적절하지 않는 경우 --> 다른 모델 사용  ==> XGBoost LightGBM (데이터량 대, 튜닝어렵다)






























