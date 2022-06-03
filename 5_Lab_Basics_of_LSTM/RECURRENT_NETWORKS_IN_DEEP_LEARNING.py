#RECURRENT NETWORKS IN DEEP LEARNING
'''
Lets first create a tiny LSTM network sample to understand the architecture of LSTM networks.

We need to import the necessary modules for our code. We need numpy and tensorflow, obviously. 
Additionally, we can import directly the tensorflow.contrib.rnn model, which includes the function for building RNNs.
'''
import numpy as np
import tensorflow as tf
sess = tf.Session()

'''
We want to create a network that has only one LSTM cell. We have to pass 2 elements to LSTM, the prv_output and prv_state, so called, h and c. 
Therefore, we initialize a state vector, state. Here, state is a tuple with 2 elements, each one is of size [1 x 4], 
one for passing prv_output to next time step, and another for passing the prv_state to next time stamp.
'''
LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
state

#Let define a sample input. In this example, batch_size = 1, and seq_len = 6:
sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))

#Now, we can pass the input to lstm_cell, and check the new state:
with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print (sess.run(state_new))

#As we can see, the states has 2 parts, the new state c, and also the output h. Lets check the output again:
print (sess.run(output))

#Stacked LSTM-----------------------------------------------------------------------------------------------------
'''
What about if we want to have a RNN with stacked LSTM? For example, a 2-layer LSTM. In this case, the output of the first layer will become the input of the second.
'''
#Lets start with a new session:
sess = tf.Session()

input_dim = 6

#Lets create the stacked LSTM cell:
cells = []

#Creating the first layer LTSM cell.
LSTM_CELL_SIZE_1 = 4 #4 hidden nodes
cell1 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

#Creating the second layer LTSM cell.
LSTM_CELL_SIZE_2 = 5 #5 hidden nodes
cell2 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

#To create a multi-layer LTSM we use the tf.contrib.rnnMultiRNNCell function, 
#it takes in multiple single layer LTSM cells to create a multilayer stacked LTSM model.
stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

#Now we can create the RNN from stacked_lstm:
# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)

'''
Lets say the input sequence length is 3, and the dimensionality of the inputs is 6. 
The input should be a Tensor of shape: [batch_size, max_time, dimension], in our case it would be (2, 3, 6)
'''
#Batch size x time steps x features.
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input

#we can now send our input to network, and check the output:
output

sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})