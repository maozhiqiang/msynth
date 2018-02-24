from utils import *


class TeacherWavenet(object):

    def __init__(self,
                 filter_width,
                 hidden_units,
                 output_classes,
                 training,
                 padding="VALID",
                 num_layers=4,
                 num_stages=2):

        # Eventually replace with variable shape
        inputs = tf.placeholder(tf.float32, shape=[1, 1, 1024, 1])

        # Eventually replace targets by placeholder and also modify 256 to get mu
        targets = tf.placeholder(tf.float32, shape=[1, 1, 1024, output_classes])

        h = dilated_conv(inputs, "start_conv", filter_width, hidden_units, trainable=training)

        skip = dilated_conv(inputs, "start_skip", 1, hidden_units, trainable=training)

        # test with 128 filters per residual and skip layer
        for i in range(num_layers):
            name = "dilated_conv_{0}".format(i)

            lay_ = dilated_conv(h, name, filter_width, hidden_units, 2 ** (i % num_stages), trainable=training)

            # modify to have gated acitvation as in wavenet (i.e. create appropraiate function)
            tanh = dilated_conv(lay_, "tanh_{0}".format(i), filter_width, hidden_units, 1, trainable=training)
            tanh = tf.tanh(tanh)

            sig = dilated_conv(lay_, "sig_{0}".format(i), filter_width, hidden_units, 1, trainable=training)
            sig = tf.sigmoid(sig)

            lay_ = tanh * sig

            # 1x1 convolution
            lay_ = dilated_conv(lay_, "1x1_{0}".format(i), 1, hidden_units, trainable=training)

            # residual output
            h += lay_

            # skip output
            skip += lay_

        # add final layer to skips
        skip += h

        skip = tf.nn.relu(skip)

        outputs = dilated_conv(skip,
                               "1x1_{0}".format(num_layers+1),
                               filter_width,
                               output_classes,
                               trainable=training)

        outputs = tf.nn.relu(outputs)

        # logits of the prob. Can recover prob with sig(outputs)
        outputs = dilated_conv(outputs,
                               "1x1_{0}".format(num_layers+2),
                               filter_width,
                               output_classes,
                               trainable=training)

        # Compute cross-entropy // modify to use KL divergence??
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs,
                                                       labels=targets)

        loss = tf.reduce_mean(loss)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

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
