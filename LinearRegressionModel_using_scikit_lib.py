
# coding: utf-8

# In[6]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
 
data = pd.read_csv('C:\Bansari\GitRepositories\LinearRegressionModel\Weight_Height_LinearRegression_DataSet.csv')
data.head()

X = data['Height'].values
Y = data['Weight'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)

m = len(X)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((m, 1))

# Creating a Model
reg = LinearRegression()

# Fitting traing data
reg = reg.fit(X, Y)

# Y Prediction
Y_pred = reg.predict(X)

# R2 score calculation
r2_score = reg.score(X, Y)

print('R2: ' + str(r2_score))
print('If person\'s height is 166cm, predicted weight of that person will be: ' + str(reg.predict(166)))


# In[ ]:



