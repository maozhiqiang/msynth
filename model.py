from utils import *


class TeacherWavenet(object):

    def __init__(self,
                 inputs,
                 targets,
                 filter_width,
                 hidden_units,
                 output_classes=2,
                 training=True,
                 padding="VALID",
                 num_layers=12,
                 num_stages=4):

        assert output_classes % 2 == 0 and hidden_units % 2 == 0

        h = dilated_conv(inputs, "start_conv", filter_width, hidden_units, trainable=training)

        skip = dilated_conv(inputs, "start_skip", 1, hidden_units, trainable=training)

        # test with 128 filters per residual and skip layer
        for i in range(num_layers):
            name = "dilated_conv_{0}".format(i)

            lay_ = dilated_conv(h, name, filter_width, hidden_units, 2 ** (i % num_stages), trainable=training)

            # modify to have gated acitvation as in wavenet (i.e. create appropraiate function)
            lay_ = dilated_conv(lay_, "tanh_sig_{0}".format(i), filter_width, hidden_units, 1, trainable=training)

            mid = int(hidden_units / 2)

            tanh = tf.tanh(lay_[:, :, :, :mid])

            sig = tf.sigmoid(lay_[:, :, :, mid:])

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
        params = dilated_conv(skip,
                              "1x1_{0}".format(num_layers+2),
                              filter_width,
                              output_classes,
                              trainable=training)

        mid = int(output_classes / 2)

        location = params[:, :, :, :mid]

        # ensure that the scale parameter is > 0
        scale = tf.nn.relu(params[:, :, :, mid:]) + tf.log(1. + tf.exp(-tf.abs(params[:, :, :, mid:])))

        logits = log_mixt(inputs, location, scale)

        log_probs = -tf.log(1. + tf.exp(-tf.abs(logits)))

        log_probs = tf.where(logits <= 0., log_probs + logits, log_probs)

        # Compute cross-entropy. Uses hard prob. Modify to allow soft prob also
        loss = tf.reduce_mean(tf.reduce_sum(-log_probs, 2)) if training else None

        train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss) if training else None

        self.inputs = inputs
        self.targets = targets
        self.logits = logits
        self.loss = loss
        self.train_step = train_step
        self.scale = scale
        self.location = location
        self.log_probs = log_probs

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
                 output_classes,
                 training=True,
                 padding="VALID",
                 num_layers=4,
                 num_stages=2):

        assert hidden_units % 2 == 0 and output_classes % 2 == 0

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
            lay_ = dilated_conv(lay_,
                                "flow_{0}_tanh_sig_{1}".format(flow, i),
                                filter_width,
                                hidden_units,
                                1,
                                trainable=training)

            mid = int(hidden_units / 2)

            tanh = tf.tanh(lay_[:, :, :, :mid])

            sig = tf.sigmoid(lay_[:, :, :, mid:])

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

        params = dilated_conv(h,
                               "flow_{0}_1x1_{1}".format(flow, num_layers + 1),
                               filter_width,
                               output_classes,
                               trainable=training)

        params = tf.nn.relu(params)

        # output mean and standard deviation for each sample. The 2 values of dimension 4
        params = dilated_conv(params,
                               "flow_{0}_1x1_{1}".format(flow, num_layers + 2),
                               filter_width,
                               output_classes,
                               trainable=training)

        mid = int(output_classes / 2)

        location = params[:, :, :, :mid] + 30.

        # ensure that the scale paramters are > 0
        scale = tf.nn.relu(params[:, :, :, mid:]) + tf.log(1. + tf.exp(-tf.abs(params[:, :, :, mid:])))

        self.inputs = inputs
        self.location = location
        self.scale = scale


# this will implement the Inverse autoregressive flow using StudentWavenetComp's
# will probably include encoder to provide additional input to the IAF such as melody
class IAF(object):

    def __init__(self,
                 inputs,
                 flows,
                 filter_width,
                 hidden_units,
                 output_classes,
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
                                       output_classes=output_classes,
                                       training=training)

            flow = block.scale * flow + block.location

            location = location * block.scale + block.location
            scale = tf.abs(block.scale * scale)

        # obtain the log probabilities for each sample
        logits = log_mixt(flow, location, scale)

        # regenerate the Teacher Wavenet and optimize
        teacher = TeacherWavenet(inputs=flow,
                                 targets=None,
                                 filter_width=3,
                                 hidden_units=64,
                                 output_classes=output_classes,
                                 training=False)

        # obtain non trainable weights
        restore = [weight for weight in tf.global_variables() if weight not in tf.trainable_variables()]

        # use the KL divergence instead and feed it the teacher probs output
        loss = kl_div(teacher.log_probs, scale) if training else None

        train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss) if training else None

        self.inputs = inputs
        self.outputs = discretize(flow, 65535.)
        self.loss = loss
        self.train_step = train_step
        self.params = [location, scale]
        self.test = teacher.log_probs
        self.teach_logits = teacher.logits
        self.restore = restore

