
# coding: utf-8

# # Import Library and Load MNIST Data


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# # Define HyperParameters and Initialize them
learning_rate = 0.0001
epochs = 10
batch_size = 50


# # Declare the training data placeholders
# 
# Input x - for 28 x 28 pixels = 784 - this is the flattened image data that we get from mnist.
# 
# We will need to reshape the the data into no_of_training_samples x width x height x channels.
x = tf.placeholder(tf.float32,shape=[None,784])
# Channel is 1 since mnist data is in grayscale
x_reshaped = tf.reshape(x,shape=[-1,28,28,1])
y = tf.placeholder(tf.float32,shape=[None,10])


# # Define function that will return the Conv Layer Unit. 
# This will have Conv Layer followed with relu activation. MaxPooling will be applied after that to subsample. 
def conv_layer_unit(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    
    #The format that the conv2d() function receives for the filter is: [filter_height, filter_width, in_channels, out_channels]
    weight_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    #Initialize weight and shape for the inputs
    weights = tf.Variable(tf.truncated_normal(weight_shape,stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]),name=name+'_b')
    
    #Setup the convolutional Layer. Note first and last should be 1 and between 2 are x and y stride
    stride = [1,1,1,1]
    outlayer = tf.nn.conv2d(input=input_data,filter=weights,strides=stride,padding='SAME')
    
    #Add the bias
    outlayer += bias
    
    #Apply the relu
    outlayer = tf.nn.relu(outlayer)
    
    #Apply the maxpooling
    ksize = [1, pool_shape[0],pool_shape[1],1]
    #Stride of 2 to downsample
    stride = [1,2,2,1]
    outlayer = tf.nn.max_pool(outlayer,ksize=ksize,strides=stride,padding='SAME')
    
    #return the layer unit
    return outlayer


# # Create Convolutional Layers
layer1 = conv_layer_unit(input_data=x_reshaped,num_input_channels=1,num_filters=32,filter_shape=[5,5],pool_shape=[2,2],name='layer1')
layer2 = conv_layer_unit(input_data=layer1,num_input_channels=32,num_filters=64,filter_shape=[5,5],pool_shape=[2,2],name='layer2')


# # The fully connected layer
# 
# we have to flatten out the output from the final convolutional layer.  It is now a 7×7 grid of nodes with 64 channels, which equates to 3136 nodes per training sample.  We can use tf.reshape() to do what we need:
# 
# ***Note: Please see the network image for correct shapes***
# 

# Fully Connected Layer 1
fc = tf.reshape(layer2,shape=[-1,7*7*64])
wd1 = tf.Variable(tf.truncated_normal([7*7*64,1000],stddev=0.03), name='fc_wd1')
bd1 = tf.Variable(tf.truncated_normal([1000]), name='fc_bd1')
fc_layer1 = tf.matmul(fc,wd1) + bd1
fc_layer1 = tf.nn.relu(fc_layer1)

# Fully Connected Layer 2
wd2 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.03), name='fc_wd2')
bd2 = tf.Variable(tf.truncated_normal([10]), name='fc_bd2')
fc_layer2 = tf.matmul(fc_layer1,wd2) + bd2
y_ = tf.nn.softmax(fc_layer2)


# # The cross-entropy cost function
# 
# TensorFlow provides a handy function which applies soft-max followed by cross-entropy loss:
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y))


# # Trainer
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, cost = sess.run([optimiser, cross_entropy],feed_dict={x: batch_x, y: batch_y})
            avg_cost += cost/total_batch
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
    
    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

