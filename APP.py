from flask import Flask,render_template, request
import os
import numpy as np
import pandas as pd
import pickle
app=Flask(__name__) 
model=pickle.load(open("/storage/emulated/0/Download/Bank_Fraud_Detector/best_model.pkl","rb"))
@app.route("/",methods=["GET","POST"])
def myfunc():
    response=None
    if request.method=="POST":
        Age=float(request.form["age"])
        Bank=float(request.form["bank"])
        NumTransactionsDay=float(request.form["transactions"])
        TransactionHour=float(request.form["hour"])
        TransactionDay=float(request.form["day"])
        LocationMatch=float(request.form["location"])
        DeviceKnown=float(request.form["deviceknown"])
        PrevFraudFlag=float(request.form["flag"])
        AccountAgeMonths=float(request.form["acc_age"])
        FailedAttempts=float(request.form["failed_attempts"])
        InternationalTransaction=float(request.form["international"])
        TransactionAmount=float(request.form["trans_amount"])
        AccountBalance=float(request.form["acc_balance"])
        Channel=float(request.form["channel"])
        AmountToBalanceRatio=TransactionAmount/AccountBalance
        n_TransactionAmount=np.log1p(TransactionAmount)
        n_AccountBalance=np.log1p(AccountBalance)
        if TransactionHour in [22,23,24,0,1,2,3,4]:
            UnusualTransactionHour=1
        else:
            UnusualTransactionHour=0 
        RedFlag_Attempt=PrevFraudFlag+FailedAttempts
        AmountToBalance_nRatio=n_TransactionAmount/n_AccountBalance
        features=np.array([[Age,Bank,NumTransactionsDay,TransactionHour,TransactionDay,Channel,LocationMatch,DeviceKnown,PrevFraudFlag,AccountAgeMonths,FailedAttempts,InternationalTransaction,AmountToBalanceRatio,n_TransactionAmount,n_AccountBalance,UnusualTransactionHour,RedFlag_Attempt,AmountToBalance_nRatio]])
        k=model.predict(features)[0]
        if k==0:
            response="Not Fraud(Clean)"
        else:
            response="Fraud(Red Flag)"
    return render_template("DESIGN.html",response=response)
if __name__==("__main__"):
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True) 
