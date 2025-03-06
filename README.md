# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4..Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Alan Samuel Vedanayagam
RegisterNumber:  212223040012
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("/content/50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
*/


## Output:
![Screenshot 2025-03-06 200516](https://github.com/user-attachments/assets/ec48988a-3742-4823-8220-44125feff59c)
![Screenshot 2025-03-06 200547](https://github.com/user-attachments/assets/3fcec8f3-5491-4fda-a197-a06fbab563cc)
![Screenshot 2025-03-06 200738](https://github.com/user-attachments/assets/0ecbde5b-c37a-4af1-ab45-0c450f099987)
![Screenshot 2025-03-06 200806](https://github.com/user-attachments/assets/5ba78f5f-9796-40b8-846d-1e599a76f08f)
![Screenshot 2025-03-06 200846](https://github.com/user-attachments/assets/8eef403d-d148-4127-8afa-5a24bf477f79)
![Screenshot 2025-03-06 200911](https://github.com/user-attachments/assets/8025d7ee-9dc8-411a-9f2e-878b4faad160)
![Screenshot 2025-03-06 200940](https://github.com/user-attachments/assets/728016d6-df01-4d3e-afd1-6a26db3e57df)









## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
