#RECOMMENDATION SYSTEM WITH A RESTRICTED BOLTZMANN MACHINE
#Acquiring the Data---------------------------------------------------------

#To start, we need to download the data we are going to use for our system. 
#The datasets we are going to use were acquired by GroupLens and contain movies, users and movie ratings by these users.

#After downloading the data, we will extract the datasets to a directory that is easily accessible.

!wget -O ./data/moviedataset.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip -o ./data/moviedataset.zip -d ./data

#With the datasets in place, let's now import the necessary libraries. 
#We will be using Tensorflow and Numpy together to model and initialize our Restricted Boltzmann Machine and Pandas to manipulate our datasets. 
#To import these libraries

#Tensorflow library. Used to implement machine learning models
import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Dataframe manipulation library
import pandas as pd
#Graph plotting library
import matplotlib.pyplot as plt
%matplotlib inline

#Loading in the Data--------------------------------------------------------------
#Loading in the movies dataset
movies_df = pd.read_csv('./data/ml-1m/movies.dat', sep='::', header=None, engine='python')
movies_df.head()

#We can do the same for the ratings.dat file:
#Loading in the ratings dataset
ratings_df = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', header=None, engine='python')
ratings_df.head()

'''
So our movies_df variable contains a dataframe that stores a movie's unique ID number, title and genres, 
while our ratings_df variable stores a unique User ID number, a movie's ID that the user has watched, 
the user's rating to said movie and when the user rated that movie.
'''
#Let's now rename the columns in these dataframes so we can better convey their data more intuitively:
movies_df.columns = ['MovieID', 'Title', 'Genres']
movies_df.head()

#And our final ratings_df:
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings_df.head()

#The Restricted Boltzmann Machine model-----------------------------------------------------------------------
#Formatting the Data--------------------------------------------------------------------
#First let's see how many movies we have and see if the movie ID's correspond with that value:
len(movies_df)

'''
Now, we can start formatting the data into input for the RBM. 
We're going to store the normalized users ratings into as a matrix of user-rating called trX, and normalize the values.
'''
user_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
user_rating_df.head()

#Lets normalize it now:
norm_user_rating_df = user_rating_df.fillna(0) / 5.0
trX = norm_user_rating_df.values
trX[0:5]

#Setting the Model's Parameters---------------------------------------------------------------------------------------
'''
Next, let's start building our RBM with TensorFlow. We'll begin by first determining the number of neurons 
in the hidden layers and then creating placeholder variables for storing our visible layer biases, hidden layer biases 
and weights that connects the hidden layer with the visible layer. 
We will be arbitrarily setting the number of neurons in the hidden layers to 20. 
You can freely set this value to any number you want since each neuron in the hidden layer will end up learning a feature.
'''
hiddenUnits = 20
visibleUnits =  len(user_rating_df.columns)
vb = tf.placeholder("float", [visibleUnits]) #Number of unique movies
hb = tf.placeholder("float", [hiddenUnits]) #Number of features we're going to learn
W = tf.placeholder("float", [visibleUnits, hiddenUnits])

'''
We then move on to creating the visible and hidden layer units and setting their activation functions. 
In this case, we will be using the tf.sigmoid and tf.relu functions as nonlinear activations since it is commonly used in RBM's.
'''
#Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
#Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

#Now we set the RBM training parameters and functions.
#Learning rate
alpha = 1.0
#Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
#Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
#Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

#And set the error function, which in this case will be the Mean Absolute Error Function.
err = v0 - v1
err_sum = tf.reduce_mean(err * err)

#We also have to initialize our variables. Thankfully, NumPy has a handy ,code>zeros function for this. We use it like so:
#Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)
#Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)
#Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)
#Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Now we train the RBM with 15 epochs with each epoch using 10 batches with size 100. After training, we print out a graph with the error by epoch.
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_nb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print (errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

#Recommendation------------------------------------------------------------------------------------------------
'''
We can now predict movies that an arbitrarily selected user might like. 
This can be accomplished by feeding in the user's watched movie preferences into the RBM and then reconstructing the input. 
The values that the RBM gives us will attempt to estimate the user's preferences for movies that he hasn't watched based on 
the preferences of the users that the RBM was trained on.
'''
#Lets first select a User ID of our mock user:
mock_user_id = 215

#Selecting the input user
inputUser = trX[mock_user_id-1].reshape(1, -1)
inputUser[0:5]

#Feeding in the user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={ v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb})
print(rec)

#We can then list the 20 most recommended movies for our mock user by sorting it by their scores given by our model.
scored_movies_df_mock = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = rec[0])
scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20)

#So, how to recommend the movies that the user has not watched yet? 
#Now, we can find all the movies that our mock user has watched before:
movies_df_mock = ratings_df[ratings_df['UserID'] == mock_user_id]
movies_df_mock.head()

#In the next cell, we merge all the movies that our mock users has watched with the predicted scores based on his historical data:
#Merging movies_df with ratings_df by MovieID
merged_df_mock = scored_movies_df_mock.merge(movies_df_mock, on='MovieID', how='outer')

#lets sort it and take a look at the first 20 rows:
merged_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20)

