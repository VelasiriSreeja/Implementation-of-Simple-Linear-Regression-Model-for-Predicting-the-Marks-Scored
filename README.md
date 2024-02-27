# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.
. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
```
Developed by: v.sreeja
RegisterNumber:  212222230169
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()


df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,regressor.predict(x_test),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

![Screenshot 2024-02-27 085913](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/0bd34387-2535-4e1a-aa79-738b4397bdd4)


![Screenshot 2024-02-27 085936](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/798f0f88-c225-4dd6-b316-eacdea5fba3e)


![Screenshot 2024-02-27 090001](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/0d4bbb61-d86a-443f-9cc2-6827ed356adf)


![Screenshot 2024-02-27 090021](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/c9249934-ac2a-4f91-b4ab-f46f327b10f0)


![Screenshot 2024-02-27 090046](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/6fad4a6f-bd90-4510-ac7b-1a3571b6d57e)


![1j](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/e1ef005e-ba16-4073-ab89-a88cc14687d1)


![Screenshot 2024-02-27 090141](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/14038c33-cd4f-455e-b8d0-3de255364244)


![Screenshot 2024-02-27 090158](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/5a7a0b32-6684-4ce1-96ad-71b9ba9f6061)


![1q](https://github.com/VelasiriSreeja/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118344328/de2ecb53-e301-40dd-8418-3d1bacd2af8c)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
