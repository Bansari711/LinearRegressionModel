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
regressor = LinearRegression()

# Fitting traing data
regressor = regressor.fit(X, Y)

# Y Prediction
Y_pred = regressor.predict(X)

# R2 score calculation
r2_score = regressor.score(X, Y)

plt.plot(X, regressor.predict(X), color='#58b970', label='Regression Line (Predicted data)')
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot (Actual data)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()

print('R2: ' + str(r2_score))
print('If person\'s height is 154cm, predicted weight of that person will be: ' + str(regressor.predict(154)))


