from utils import *


class TeacherWavenet(object):

    def __init__(self,
                 inputs,
                 targets,
                 filter_width,
                 hidden_units,
                 output_classes,
                 training,
                 padding="VALID",
                 num_layers=30,
                 num_stages=10):

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

        skip = dilated_conv(skip,
                            "1x1_{0}".format(num_layers+1),
                            filter_width,
                            output_classes,
                            trainable=training)

        skip = tf.nn.relu(skip)

        # logits of the prob. Can recover prob with sig(logits)
        logits = dilated_conv(skip,
                              "1x1_{0}".format(num_layers+2),
                              filter_width,
                              output_classes,
                              trainable=training)

        # Compute cross-entropy // modify to use KL divergence??
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                       labels=targets) if training else None

        loss = tf.reduce_mean(loss) if training else None

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) if training else None

        self.inputs = inputs
        self.targets = targets
        self.logits = logits
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


# Studente Wavenet components used to create an IAF object (the final student Wavenet)
class StudentWavenetComp(object):

    def __init__(self,
                 inputs,
                 flow,
                 filter_width,
                 hidden_units,
                 training,
                 padding="VALID",
                 num_layers=10,
                 num_stages=10):

        h = dilated_conv(inputs,
                         "flow_{0}_start_conv".format(flow),
                         filter_width, hidden_units,
                         trainable=training)

        # test with 128 filters per residual and skip layer
        for i in range(num_layers):
            name = "flow_{0}_dilated_conv_{1}".format(flow, i)

            lay_ = dilated_conv(h,
                                name,
                                filter_width,
                                hidden_units,
                                2 ** (i % num_stages),
                                trainable=training)

            # modify to have gated acitvation as in wavenet (i.e. create appropraiate function)
            tanh = dilated_conv(lay_,
                                "flow_{0}_tanh_{1}".format(flow, i),
                                filter_width,
                                hidden_units,
                                1,
                                trainable=training)
            tanh = tf.tanh(tanh)

            sig = dilated_conv(lay_,
                               "flow_{0}_sig_{1}".format(flow, i),
                               filter_width,
                               hidden_units,
                               1,
                               trainable=training)
            sig = tf.sigmoid(sig)

            lay_ = tanh * sig

            # 1x1 convolution
            lay_ = dilated_conv(lay_,
                                "flow_{0}_1x1_{1}".format(flow, i),
                                1,
                                hidden_units,
                                trainable=training)

            # residual output
            h += lay_

        h = tf.nn.relu(h)

        outputs = dilated_conv(h,
                               "flow_{0}_1x1_{1}".format(flow, num_layers + 1),
                               filter_width,
                               2,
                               trainable=training)

        outputs = tf.nn.relu(outputs)

        # output mean and standard deviation for each sample. The 2 values of dimension 4
        outputs = dilated_conv(outputs,
                               "flow_{0}_1x1_{1}".format(flow, num_layers + 2),
                               filter_width,
                               2,
                               trainable=training)

        self.inputs = inputs
        self.location = outputs[:, :, :, :1]
        self.scale = tf.abs(outputs[:, :, :, 1:])  # ????????? leave this or not ?????????


# this will implement the Inverse autoregressive flow using StudentWavenetComp's
# will probably include encoder to provide additional input to the IAF such as melody
class IAF(object):

    def __init__(self,
                 inputs,
                 flows,
                 filter_width,
                 hidden_units,
                 training):

        flow = inputs

        location = tf.zeros(shape=inputs.shape, dtype=tf.float32)

        scale = tf.ones(shape=inputs.shape, dtype=tf.float32)

        for i in range(flows):

            # Double check for the number of layers in individual flows
            # Need to have different layers for different flows
            block = StudentWavenetComp(inputs=flow,
                                       flow=i,
                                       filter_width=filter_width,
                                       hidden_units=hidden_units,
                                       training=training)

            flow = block.scale * flow + block.location

            location = location * block.scale + block.location
            scale = block.scale * scale

        # obtain the probabilities for each sample
        probs = disc_log_mixt(flow, [location, scale], 65535., 0.)

        # regenerate the Teacher Wavenet and optimize
        teacher = TeacherWavenet(inputs=flow,
                                 targets=None,
                                 filter_width=3,
                                 hidden_units=64,
                                 output_classes=1,
                                 training=False)

        # obtain non trainable weights
        restore = [weight for weight in tf.global_variables() if weight not in tf.trainable_variables()]

        # use the KL divergence instead and feed it the teacher probs output
        loss = kull_leib(z=inputs,
                         s_probs=probs,
                         t_probs=tf.sigmoid(teacher.logits),
                         location=location,
                         scale=scale) if training else None

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) if training else None

        self.inputs = inputs
        self.outputs = flow
        self.loss = loss
        self.train_step = train_step
        self.params = [location, scale]
        self.probs = probs
        self.test = tf.sigmoid(teacher.logits)
        self.restore = restore

