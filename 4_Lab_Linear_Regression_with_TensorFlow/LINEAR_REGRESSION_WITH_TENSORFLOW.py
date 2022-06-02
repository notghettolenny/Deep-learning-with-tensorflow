#LINEAR REGRESSION WITH TENSORFLOW
#Let's first import the required packages:
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)

#Let's define the independent variable:
X = np.arange(0.0, 5.0, 0.1)
X

##You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

Y= a * X + b 

plt.plot(X, Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

'''
A simple example of a linear function can help us understand the basic mechanism behind TensorFlow.

For the first part we will use a sample dataset, and then we'll use TensorFlow to adjust and get the right parameters. 
We download a dataset that is related to fuel consumption and Carbon dioxide emission of cars. 
'''
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

#Understanding the Data-------------------------------------------------------------------------------------------
'''
FuelConsumption.csv:
We have downloaded a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. Dataset source

    MODELYEAR e.g. 2014
    MAKE e.g. Acura
    MODEL e.g. ILX
    VEHICLE CLASS e.g. SUV
    ENGINE SIZE e.g. 4.7
    CYLINDERS e.g 6
    TRANSMISSION e.g. A6
    FUEL CONSUMPTION in CITY(L/100 km) e.g. 9.9
    FUEL CONSUMPTION in HWY (L/100 km) e.g. 8.9
    FUEL CONSUMPTION COMB (L/100 km) e.g. 9.2
    CO2 EMISSIONS (g/km) e.g. 182 --> low --> 0
'''
df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

#Lets say we want to use linear regression to predict Co2Emission of cars based on their engine size. 
#So, lets define X and Y value for the linear regression, that is, train_x and train_y:
train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

#First, we initialize the variables a and b, with any random guess, and then we define the linear function:
a = tf.Variable(20.0)
b = tf.Variable(30.2)
y = a * train_x + b

'''
Now, we are going to define a loss function for our regression, so we can train our model to better fit our data. 
In a linear regression, we minimize the squared error of the difference between the predicted values(obtained from the equation) 
and the target values (the data that we have). In other words we want to minimize the square of the predicted values minus the target value. 
So we define the equation to be minimized as loss.

To find value of our loss, we use tf.reduce_mean(). This function finds the mean of a multidimensional tensor, and the result can have a different dimension.
'''
loss = tf.reduce_mean(tf.square(y - train_y))

'''
Then, we define the optimizer method. The gradient Descent optimizer takes in parameter: learning rate, which corresponds to 
the speed with which the optimizer should learn; there are pros and cons for increasing the learning-rate parameter, with a high 
learning rate the training model converges quickly, but there is a risk that a high learning rate causes instability and the model will not converge. 
Please feel free to make changes to learning parameter and check its effect. On the other hand decreasing the learning rate might 
reduce the convergence speed, but it would increase the chance of converging to a solution. You should note that the solution might not be 
a global optimal solution as there is a chance that the optimizer will get stuck in a local optimal solution. Please review other material for further 
information on the optimization. Here we will use a simple gradient descent with a learning rate of 0.05: 
'''
optimizer = tf.train.GradientDescentOptimizer(0.05)

'''
Now we will define the training method of our graph, what method we will use for minimize the loss? 
We will use the .minimize() which will minimize the error function of our optimizer, resulting in a better model.
'''
train = optimizer.minimize(loss)

#Don't forget to initialize the variables before executing a graph:
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Now we are ready to start the optimization and run the graph:
loss_values = []
train_data = []
for step in range(100):
    _, loss_val, a_val, b_val = sess.run([train, loss, a, b])
    loss_values.append(loss_val)
    if step % 5 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])

#Lets plot the loss values to see how it has changed during the training:
plt.plot(loss_values, 'ro')

#Lets visualize how the coefficient and intercept of line has changed to fit the data:
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(train_x, train_y, 'ro')


green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()

