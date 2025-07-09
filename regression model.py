import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('/Users/palammysurareddy/Downloads/Salary_Data - Salary_Data.csv')

x = dataset.iloc[:, :-1]  
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) 

y_pred = regressor.predict(x_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)


plt.scatter(x_test, y_test, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)


y_12 = (m_slope*12) + c_intercept
print(y_12)


y_20 = (m_slope*20) + c_intercept
print(y_20)

dataset.mean()



dataset.std()

dataset.var()


from scipy.stats import variation
variation(dataset.values) 

variation(dataset['Salary']) 


dataset.corr()

dataset.skew()

dataset.sem()

# inferential stats
# z-score 
import scipy.stats as stats
dataset.apply(stats.zscore)

# sum of squer regresso ( SSR )
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)


# SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# SST 
mean_total = np.mean(dataset.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


# R2 SQUER 

r_square = 1 - (SSR/SST)
r_square


print (regressor)

bias = regressor.score(x_train,y_train)
print(bias)

variance  = regressor.score(x_test,y_test)
print(variance)




import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")



