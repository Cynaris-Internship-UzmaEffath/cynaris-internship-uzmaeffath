import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
data= fetch_california_housing(as_frame=True)
X=data.data
y=data.target
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.02,random_state=42)
print("\nTraining samples:",len(X_train))
print("\nTesting samples:",len(X_test))
model=LinearRegression()
model.fit(X_train,y_train)
print("\n Linear Regression coefficients:\n")
print(pd.Series(model.coef_,index=X.columns))
print("\n Intercept:\n",model.intercept_)
predictions=model.predict(X_test)
mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,predictions)
r2=r2_score(y_test,predictions)
print("\n Linear Regression Results:\n\n")
print("\nMSE:\n",round(mse,4))
print("\nRMSE:\n",round(rmse,4))
print("\nMSAE:\n",round(mae,4))
print("\nR2:\n",round(r2,4))