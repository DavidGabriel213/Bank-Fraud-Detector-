import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/storage/emulated/0/Download/nigerian_bank_fraud_messy.csv")
df=df.drop_duplicates()
df["TransactionID"]=df["TransactionID"].astype(str).str.strip()
df["Age"]=df["Age"].astype(str).str.replace("yrs","").str.replace("years","").str.strip()
df["Age"]=pd.to_numeric(df["Age"],errors="coerce")
df["Age"]=df["Age"].apply(lambda x: np.nan if x>110 else x)
df["Age"]=(df["Age"].fillna(df["Age"].median())).astype(int)
df["Gender"]=df["Gender"].astype(str).str.capitalize().str.strip()
gender_corrector={"M":"Male","F":"Female"}
df["Gender"]=df["Gender"].replace(gender_corrector)
df["State"]=df["State"].astype(str).str.strip()
df["Bank"]=df["Bank"].astype(str).str.strip()
df["AccountType"]=df["AccountType"].astype(str).str.capitalize().str.strip()
AccountType_corrector={"Cur":"Current","Sav":"Savings","Dom":"Domiciliary","S":"Savings","C":"Current"}
df["AccountType"]=df["AccountType"].replace(AccountType_corrector)
df["TransactionAmount"]=df["TransactionAmount"].astype(str).str.replace("\"","").str.replace("NGN","").str.replace("-","").str.replace("\u20a6","").str.replace(",","").str.strip()
df["TransactionAmount"]=pd.to_numeric(df["TransactionAmount"],errors="coerce")
max1=(df["TransactionAmount"].quantile(0.75)+1.5*(df["TransactionAmount"].quantile(0.75)-df["TransactionAmount"].quantile(0.25))).round(2)
min1=(df["TransactionAmount"].quantile(0.25)-1.5*(df["TransactionAmount"].quantile(0.75)-df["TransactionAmount"].quantile(0.25))).round(2)
df["TransactionAmount"]=df["TransactionAmount"].clip(min1,max1)
df["TransactionAmount"]=(df["TransactionAmount"].fillna(df.groupby(["State","Gender"])["TransactionAmount"].transform("mean"))).round(2)

df["AccountBalance"]=df["AccountBalance"].astype(str).str.replace("\"","").str.replace("NGN","").str.replace("-","").str.replace("\u20a6","").str.replace(",","").str.strip()
df["AccountBalance"]=pd.to_numeric(df["AccountBalance"],errors="coerce")
max1=(df["AccountBalance"].quantile(0.75)+1.5*(df["AccountBalance"].quantile(0.75)-df["AccountBalance"].quantile(0.25))).round(2)
min1=(df["AccountBalance"].quantile(0.25)-1.5*(df["AccountBalance"].quantile(0.75)-df["AccountBalance"].quantile(0.25))).round(2)
df["AccountBalance"]=df["AccountBalance"].clip(min1,max1)
df["AccountBalance"]=(df["AccountBalance"].fillna(df.groupby(["State","Gender"])["AccountBalance"].transform("mean"))).round(2)
df["NumTransactionsDay"]=np.abs(df["NumTransactionsDay"])
df["TransactionHour"]=df["TransactionHour"].astype(str).str.strip()
def transactionHour_corrector(c):
    if ":" in c:
        return c[:c.index(":")]
    elif "AM" in c:
        return c.replace("AM","").strip()
    elif "PM" in c:
       h = float(c.replace("PM","").strip())
       return str(h+12 if h < 12 else h)
    else:
        return c
df["TransactionHour"]=df["TransactionHour"].apply(lambda x: transactionHour_corrector(x))
df["TransactionHour"]=pd.to_numeric(df["TransactionHour"],errors="coerce")
df["TransactionHour"]=df["TransactionHour"].apply(lambda x: 0 if x==25 else x)
df["TransactionHour"]=(df["TransactionHour"].fillna(df["TransactionHour"].median())).astype(int)
df["TransactionDay"]=df["TransactionDay"].astype(str).str.strip()
df["Channel"]=df["Channel"].astype(str).str.capitalize().str.strip()
channel_corrector={"Web":"Online","Nan":np.nan}
df["Channel"]=df["Channel"].replace(channel_corrector)
df["LocationMatch"]=df["LocationMatch"].astype(str).str.capitalize().str.strip()
location_corrector={"0":"No","1":"Yes","True":"Yes","False":"No","Y":"Yes","N":"No"}
df["LocationMatch"]=df["LocationMatch"].replace(location_corrector)
df["DeviceKnown"]=df["DeviceKnown"].astype(str).str.capitalize().str.replace("0","No").str.replace("1","Yes").str.replace("Unknown","No").str.strip()
df["AccountAgeMonths"]=df["AccountAgeMonths"].astype(str).str.strip()
def account_age_corrector(x):
    if "years" in x:
        return str((float(x.replace("years","")))*12)
    elif "months" in  x:
        return x.replace("months","")
    elif x=="nan":
        return np.nan
    else:
        return x
df["AccountAgeMonths"]=df["AccountAgeMonths"].apply(lambda x: account_age_corrector(x))
df["AccountAgeMonths"]=pd.to_numeric(df["AccountAgeMonths"],errors="coerce")
df["AccountAgeMonths"]=df["AccountAgeMonths"].fillna(df["AccountAgeMonths"].median())
df["FailedAttempts"]=df["FailedAttempts"].apply(lambda x: x/100 if x>=100 else x)
df["FailedAttempts"]=(df["FailedAttempts"].fillna(df.groupby(["Bank","Channel"])["FailedAttempts"].transform(lambda x:x.mode()[0]))).astype(int)
df["InternationalTransaction"]=df["InternationalTransaction"].astype(str).str.capitalize().str.strip()
df["InternationalTransaction"]=df["InternationalTransaction"].str.replace("1","Yes").str.replace("0","No")
df["AmountToBalanceRatio"]=(df["TransactionAmount"]/df["AccountBalance"]).round(4)
df["IsFraud"]=df["IsFraud"].astype(str).str.capitalize().str.strip()
IsFraud_corrector={"1":"Yes","0":"No","Clean":"No","False":"No","Fraud":"Yes","True":"Yes","False":"No"}
df["IsFraud"]=df["IsFraud"].replace(IsFraud_corrector)
df.to_csv("BankFraudCleanedData.csv",index=False)
df["n_TransactionAmount"]=(np.log1p(df["TransactionAmount"])).round(4)
df["n_AccountBalance"]=(np.log1p(df["AccountBalance"])).round(4)
import seaborn as sns
df["UnusualTransactionHour"]=df["TransactionHour"].apply(lambda x: "RedFlag" if x in [22,23,24,0,1,2,3,4] else "Normal")
df["AgeTransactionsRatio"]=(df["NumTransactionsDay"]/(1+df["AccountAgeMonths"])).round(4)
df["RedFlag_Attempt"]=df["PrevFraudFlag"]+df["FailedAttempts"]
df["AmountToBalance_nRatio"]=(df["n_TransactionAmount"]/df["n_AccountBalance"]).round(4)
df.to_csv("BankFraudClean&Engineerig.csv",index=False)
fig,ax=plt.subplots(3,3,figsize=(10,10))
FraudbyState=df.groupby("State")["IsFraud"].value_counts().unstack()
FraudbyState.plot(ax=ax[0,0],kind="bar")
ax[0,0].set_title("FraudCountbyState",color="red")
ax[0,0].set_ylabel("No.ofFraud")

FraudbyStateGender=df.groupby(["State","Gender"])["IsFraud"].value_counts().unstack()
FraudbyStateGender.plot(ax=ax[0,1],kind="bar")
ax[0,1].set_title("FraudCountbyGenderState",color="red")
ax[0,1].set_ylabel("No.ofFraud")

FraudbyGender=df.groupby("Gender")["IsFraud"].value_counts().unstack()
FraudbyGender.plot(ax=ax[0,2],kind="bar")
ax[0,2].set_title("FraudCountbyGender",color="red")
ax[0,2].set_ylabel("No.ofFraud")

FraudbyAge=df.groupby("IsFraud")["Age"].mean()
FraudbyAge.plot(ax=ax[1,0],kind="bar")
ax[1,0].set_title("FraudCountbyAge(average)",color="red")
ax[1,0].set_ylabel("Age(yrs)")

FraudChannel=df.groupby("Channel")["IsFraud"].value_counts().unstack()
FraudChannel.plot(ax=ax[1,1],kind="bar")
ax[1,1].set_title("FraudCountbyChannel)",color="red")
ax[1,1].set_ylabel("No.ofFraud")

FraudInternational=df.groupby("InternationalTransaction")["IsFraud"].value_counts().unstack()
FraudInternational.plot(ax=ax[1,2],kind="bar")
ax[1,2].set_title("FraudCountbyInternationalTrctn)",color="red")
ax[1,2].set_ylabel("No.ofFraud")

FraudTransactionAmount=df.groupby("IsFraud")["TransactionAmount"].mean().round(2)
FraudTransactionAmount.plot(ax=ax[2,0],kind="bar")
ax[2,0].set_title("FraudbyAverageTrsctnAmount",color="red")
ax[2,0].set_ylabel("TransactionAmount")

FraudDay=df.groupby("TransactionDay")["IsFraud"].value_counts().unstack()
FraudDay.plot(ax=ax[2,1],kind="bar")
ax[2,1].set_title("FraudcountbyDay",color="red")
ax[2,1].set_ylabel("Fraud_count")

FraudFailAttempt=df.groupby("FailedAttempts")["IsFraud"].value_counts().unstack()
FraudFailAttempt.plot(ax=ax[2,2],kind="bar")
ax[2,2].set_title("FraudFailAttempt",color="red")
ax[2,2].set_ylabel("Fraud_count")
plt.tight_layout()
plt.show()
        
