#CONVOLUTIONAL NEURAL NETWORK APPLICATION
import tensorflow as tf
tf.__version__

#Import the MNIST dataset using TensorFlow built-in feature
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Creating an interactive section--------------------------------------------------
'''
You have two basic options when using TensorFlow to run your code:

    [Build graphs and run session] Do all the set-up and THEN execute a session to evaluate tensors and run operations (ops)
    [Interactive session] create your coding and run on the fly.

For this first part, we will use the interactive session that is more suitable for environments like Jupyter notebooks.
'''
sess = tf.InteractiveSession()

#Creating placeholders--------------------------------------------------------------
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Assigning bias and weights to null tensors-------------------------------------------
# Weight tensor
W = tf.Variable(tf.zeros([784, 10],tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10],tf.float32))

#Execute the assignment operation--------------------------------------------
# run the op initialize_all_variables using an interactive session
sess.run(tf.global_variables_initializer())

#Adding Weights and Biases to input----------------------------------------------
# mathematical operation to add weights and biases to the inputs
tf.matmul(x,W) + b

#Softmax Regression-----------------------------------------------------------------
y = tf.nn.softmax(tf.matmul(x,W) + b)

#Cost function------------------------------------------------------------------
#It is a function that is used to minimize the difference between the right answers (labels) and estimated outputs by our Network. 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Type of optimization: Gradient Descent-----------------------------------------------
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Training batches----------------------------------------------------------------------
'''
Train using minibatch Gradient Descent.

In practice, Batch Gradient Descent is not often used because is too computationally expensive. 
The good part about this method is that you have the true gradient, but with the expensive computing task of using the whole dataset in one time. 
Due to this problem, Neural Networks usually use minibatch to train.
'''
#Load 50 training examples for each training iteration   
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Test---------------------------------------------------------------------------
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

sess.close() #finish the session

#-------------------------------------------------------------------------------------
#Evaluating the final result----------------------------------------------------------------
#Starting the code
import tensorflow as tf

# finish possible remaining session
sess.close()

#Start interactive session
sess = tf.InteractiveSession()

#The MNIST data--------------------------------------------------------------------------------
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Initial parameters
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

#Input and output
#Create place holders for inputs and outputs
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

#Converting images of the data set to tensors
x_image = tf.reshape(x, [-1,28,28,1])  
x_image

#Convolutional Layer 1-------------------------------------------------------------------
'''
Defining kernel weight and bias
We define a kernel here. The Size of the filter/kernel is 5x5; Input channels is 1 (grayscale); 
and we need 32 different feature maps (here, 32 feature maps means 32 different filters are applied on each image. 
So, the output of convolution layer would be 28x28x32). 
In this step, we create a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
'''
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

#Apply the ReLU activation Function----------------------------------------------------------------
'''
In this step, we just go through all outputs convolution layer, convolve1, and wherever a negative number occurs, we swap it out for a 0. 
It is called ReLU activation Function.
Let f(x) is a ReLU activation function 𝑓(𝑥)=𝑚𝑎𝑥(0,𝑥).
'''
h_conv1 = tf.nn.relu(convolve1)

#Apply the max pooling--------------------------------------------------------------
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv1

#Convolutional Layer 2----------------------------------------------------------------------
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

#Convolve image with weight tensor and add biases.
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

#Apply the ReLU activation Function-------------------------------------------------------
h_conv2 = tf.nn.relu(convolve2)

#Apply the max pooling----------------------------------------------------------------
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv2

#Fully Connected Layer----------------------------------------------------------
#Flattening Second Layer
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])

'''
Weights and Biases between layer 2 and 3

Composition of the feature map from the last layer (7x7) multiplied by the number of feature maps (64); 1027 outputs to Softmax layer
'''
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

#Matrix Multiplication (applying weights and biases)---------------------------
fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1

#Apply the ReLU activation Function-----------------------------------
h_fc1 = tf.nn.relu(fcl)
h_fc1

#Dropout Layer, Optional phase for reducing overfitting-----------------------   ---------------------------------------
'''
It is a phase where the network "forget" some features. At each training step in a mini-batch, some units get switched off randomly 
so that it will not interact with the network. That is, it weights cannot be updated, nor affect the learning of the other network nodes. 
This can be very useful for very large neural networks to prevent overfitting.
'''
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

#Readout Layer (Softmax Layer)--------------------------------------------------------
'''
Type: Softmax, Fully Connected Layer.
Weights and Biases

In last layer, CNN takes the high-level filtered images and translate them into votes using softmax. 
Input channels: 1024 (neurons from the 3rd Layer); 10 output features
'''
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

#Matrix Multiplication (applying weights and biases)
fc=tf.matmul(layer_drop, W_fc2) + b_fc2

#Apply the Softmax activation Function-------------------------------------------
#softmax allows us to interpret the outputs of fcl4 as probabilities. So, y_conv is a tensor of probabilities.
y_CNN= tf.nn.softmax(fc)
y_CNN

#Define functions and train the model-------------------------     -------------------------------------------------
#Define the loss function
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

#Define the optimizer
'''
It is obvious that we want minimize the error of our network which is calculated by cross_entropy metric. 
To solve the problem, we have to compute gradients for the loss (which is minimizing the cross-entropy) and apply gradients to variables. 
It will be done by an optimizer: GradientDescent or Adagrad. 
'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Define prediction
#Do you want to know how many of the cases in a mini-batch has been classified correctly? lets count them.
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))

#Define accuracy
#It makes more sense to report accuracy using average of correct cases.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Run session, train
sess.run(tf.global_variables_initializer())

#If you want a fast result (it might take sometime to train it)
for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#Evaluate the model----------------------------------------------------------------------
# evaluate in batches to avoid out-of-memory issues
n_batches = mnist.test.images.shape[0] // 50
cumulative_accuracy = 0.0
for index in range(n_batches):
    batch = mnist.test.next_batch(50)
    cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("test accuracy {}".format(cumulative_accuracy / n_batches))


#Visualization
#Do you want to look at all the filters?
kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32, -1]))

!wget --output-document utils1.py http://deeplearning.net/tutorial/code/utils.py
import utils1
from utils1 import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')  

#Do you want to see the output of an image passing through first convolution layer?
import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")

ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

#What about second convolution layer?
ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

sess.close() #finish the session