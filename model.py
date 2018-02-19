from utils import *

class Wavenet(object):


    def __init__(self,
                 filter_width,
                 hidden_units,
                 output_classes,
                 padding="VALID",
                 num_layers=1,
                 num_stages=1):

        #Eventually replace with variable shape
        inputs = tf.placeholder(tf.float32, shape=[1, 1, 1024, 1])

        # Eventually replace targets by placeholder and also modify 256 to get mu
        targets = tf.placeholder(tf.float32, shape=[1, 1, 1024, 256])

        h = dilated_conv(inputs, "start_conv", filter_width, hidden_units)

        s = dilated_conv(inputs, "start_skip", 1, hidden_units)


        #test with 128 filters per residual and skip layer
        for i in range(num_layers):
            name = "dilated_conv_{0}".format(i)

            l = dilated_conv(h, name, filter_width, hidden_units, 2 ** (i % num_stages))

            # modify to have gated acitvation as in wavenet (i.e. create appropraiate function)
            t = dilated_conv(l, "tanh_{0}".format(i), filter_width, hidden_units, 1)
            t = tf.tanh(t)

            s = dilated_conv(l, "sig_{0}".format(i), filter_width, hidden_units, 1)
            s = tf.sigmoid(s)

            l = t * s

            # 1x1 convolution
            l = dilated_conv(l, "1x1_{0}".format(i), 1, hidden_units)

            # residual output
            h += l

            #skip output
            s += l


        #add final layer to skips
        s += h

        s = tf.nn.relu(s)

        outputs = dilated_conv(s,
                              "1x1_{0}".format(num_layers+1),
                              filter_width,
                              output_classes)

        outputs = tf.nn.relu(outputs)

        outputs = dilated_conv(outputs,
                              "1x1_{0}".format(num_layers+2),
                              filter_width,
                              output_classes)


        # Compute cross-entropy
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs,
                                                              labels=targets)

        loss = tf.reduce_mean(loss)

        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.loss = loss
        self.train_step = train_step



    """
    This function trains the model using an Adam Optimizer to learn the optimal weights
    
    Returns:
        Train Op
    """
    def train(self):

        step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

        return step



class IAF(object):
    pass
