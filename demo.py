import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy.random as random

#read data
#Use read_table instead of read_fwf because challenge data is set with , not tabs
dataframe = pd.read_table('challenge_dataset.txt', ',') 
x_values = dataframe[['x']]#added header
y_values = dataframe[['y']]#added header

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)



#choose a random line contained in the dataset
line_to_test = random.randint(0, dataframe.shape[0])
#predict Y for random X
predicted_value = body_reg.predict(dataframe.iloc[line_to_test]['x'])
true_value = dataframe.iloc[line_to_test]['y']
print('Predicted value : ' + str(predicted_value))
print("True value : " + str(true_value))
print("Difference : " + str(predicted_value - true_value))


#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
