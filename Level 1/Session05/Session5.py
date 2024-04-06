import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

# This is a function to draw a line from c(intercept), m(slope) (y = c + mx)
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color="red")

# Reading csv file from github repo
advertising = pd.read_csv('../input/tvmarketingcsv/tvmarketing.csv')

# Display the first 5 rows
advertising.head()

# Check the shape of the DataFrame (rows, columns)
advertising.shape

# Let's check the columns
advertising.info()

# Let's look at some statistical information about the dataframe.
advertising.describe()

# Visualise the relationship between the features and the response using scatterplots
advertising.plot(x='TV',y='Sales',kind='scatter')

X = advertising['TV']
y = advertising['Sales']

# Print the first 5 rows
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
type(X_train)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#It is a general convention in scikit-learn that observations are rows, while features are columns.
#This is needed only when you are using a single feature; in this case, 'TV'.

#Simply put, numpy.newaxis is used to increase the dimension of the existing array by one more dimension,
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()

# Fit the model using lr.fit()
lr.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
r2

advertising.plot(x='TV',y='Sales',kind='scatter')
abline(lr.coef_, lr.intercept_)