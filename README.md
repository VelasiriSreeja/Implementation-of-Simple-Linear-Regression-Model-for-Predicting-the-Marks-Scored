# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
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
![Screenshot (44)](https://user-images.githubusercontent.com/118344328/229673784-c73994cb-9e96-4774-af70-5b924d7ee59a.png)

![Screenshot (45)](https://user-images.githubusercontent.com/118344328/229674852-a891db5f-b5e2-45a9-97cf-62e244edfaf6.png)
![Screenshot (46)](https://user-images.githubusercontent.com/118344328/229674349-a230a4d9-5df8-4fb9-9473-72e746ac1442.png)
![Screenshot (48)](https://user-images.githubusercontent.com/118344328/229674401-8e682631-8315-461d-bae2-2700c82c4ac0.png)
![Screenshot (49)](https://user-images.githubusercontent.com/118344328/229674424-58a4e271-9a8e-44a5-8647-a2264674b7a5.png)
![Screenshot (51)](https://user-images.githubusercontent.com/118344328/229674454-346aa5d1-7d07-4554-84d6-0c387a88af00.png)
![Screenshot (52)](https://user-images.githubusercontent.com/118344328/229674504-86aeceea-172a-4489-94cd-326a29020859.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
