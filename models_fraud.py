import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2,figsize=(10,10))
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score,classification_report
df=pd.read_csv("/storage/emulated/0/Download/Bank_Fraud_Detector/BankFraudClean&Engineerig.csv")
le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
df["State"]=le.fit_transform(df["State"])
df["Bank"]=le.fit_transform(df["Bank"])
df["AccountType"]=le.fit_transform(df["AccountType"])
df["Channel"]=le.fit_transform(df["Channel"])
df["LocationMatch"]=le.fit_transform(df["LocationMatch"])
df["DeviceKnown"]=le.fit_transform(df["DeviceKnown"])
df["InternationalTransaction"]=le.fit_transform(df["InternationalTransaction"])
df["TransactionDay"]=le.fit_transform(df["TransactionDay"])
df["IsFraud"]=le.fit_transform(df["IsFraud"])
df["UnusualTransactionHour"]=le.fit_transform(df["UnusualTransactionHour"])
X=df[["Age","Bank","NumTransactionsDay","TransactionHour","TransactionDay","Channel","LocationMatch","DeviceKnown","PrevFraudFlag","AccountAgeMonths","FailedAttempts","InternationalTransaction","AmountToBalanceRatio","n_TransactionAmount","n_AccountBalance","UnusualTransactionHour","RedFlag_Attempt","AmountToBalance_nRatio"]]
y=df["IsFraud"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#logisticregression
model=LogisticRegression(max_iter=1000,class_weight="balanced",C=0.05)
model.fit(X_train,y_train)

y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.7
y_pred_new = (y_prob > threshold).astype(int)
accuracy=accuracy_score(y_test,y_pred_new)
report=classification_report(y_test,y_pred_new)
print(f"Accuracy if best model {accuracy:.4f}")
print(f"Classification Report for best model{report}")
cm=confusion_matrix(y_pred_new,y_test)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["No","Yes"])
disp.plot(ax=ax[0,0],cmap="Blues")
ax[0,0].set_title("LogisticRegression")
#decisiontree
model1=DecisionTreeClassifier(max_depth=7,class_weight="balanced")
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
accuracy1=accuracy_score(y_test,y_pred1)
report1=classification_report(y_test,y_pred1)
print(f"Accuracy if best model {accuracy1:.4f}")
print(f"Classification Report for best model{report1}")
cm1=confusion_matrix(y_pred1,y_test)
disp1=ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=["No","Yes"])
disp1.plot(ax=ax[0,1],cmap="Blues")
ax[0,1].set_title("DecisionTree")
#RandomForestClassifier
model2=RandomForestClassifier(n_estimators=150,random_state=42,class_weight="balanced")
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
accuracy2=accuracy_score(y_test,y_pred2)
report2=classification_report(y_test,y_pred2)
print(f"Accuracy if best model {accuracy2:.4f}")
print(f"Classification Report for best model{report2}")
cm2=confusion_matrix(y_pred2,y_test)
disp2=ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=["No","Yes"])
disp2.plot(ax=ax[1,0],cmap="Blues")
ax[1,0].set_title("RandomForest")
#finetuning best model
params={
    "C": [0.01,0.5,0.1,1],
    "class_weight":[None, "balanced"]
}
grid=GridSearchCV(
    LogisticRegression(max_iter=1000),
    params,
    scoring="f1", 
    cv=5
)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
y_pred3=grid.best_estimator_.predict(X_test)
accuracy3=accuracy_score(y_pred3,y_test)
report3=classification_report(y_test,y_pred3)
print(f"Accuracy if best model {accuracy3:.4f}")
print(f"Classification Report for best model{report3}")
cm3=confusion_matrix(y_test,y_pred3)
disp3=ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=["No","Yes"])
disp3.plot(ax=ax[1,1],cmap="Blues")
ax[1,1].set_title("LogisticFineTune")
best_model=grid.best_estimator_
df_best=pd.DataFrame({
    "feature": X.columns,
    "coef": best_model.coef_[0]
})
print(df_best.sort_values(by="coef"))
pickle.dump(best_model,open("/storage/emulated/0/Download/Bank_Fraud_Detector/best_model.pkl","wb"))
plt.show()
