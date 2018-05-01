from utils import *


class TeacherWavenet(object):

    def __init__(self,
                 inputs,
                 cond,
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

            cond_ = dilated_conv(cond, "cond_{0}".format(i), filter_width, hidden_units, 1, trainable=training)

            mid = int(hidden_units / 2)

            tanh = tf.tanh(lay_[:, :, :, :mid] + cond_[:, :, :, :mid])

            sig = tf.sigmoid(lay_[:, :, :, mid:] + cond_[:, :, :, mid:])

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

        params = dilated_conv(skip,
                              "1x1_{0}".format(num_layers+2),
                              filter_width,
                              output_classes,
                              trainable=training)

        mid = int(output_classes / 2)

        location = params[:, :, :, :mid]

        # ensure that the scale parameter is > 0
        scale = tf.abs(params[:, :, :, mid:])

        logistic = tf.contrib.distributions.Logistic(loc=location, scale=scale)

        categorical = tf.distributions.Categorical(
            probs=[1./(output_classes / 2) for _ in range(int(output_classes/2.))])

        mixture = tf.contrib.distributions.MixtureSameFamily(
            mixture_distribution=categorical,
            components_distribution=logistic
        )

        log_probs = tf.expand_dims(mixture.log_prob(tf.squeeze(inputs, -1)), -1)

        self.inputs = inputs
        self.cond = cond
        self.scale = scale
        self.location = location
        self.log_probs = log_probs
        self.mixture = mixture


# Studente Wavenet components used to create an IAF object (the final student Wavenet)
class StudentWavenetComp(object):

    def __init__(self,
                 inputs,
                 cond,
                 flow,
                 filter_width,
                 hidden_units,
                 output_classes,
                 training=True,
                 padding="VALID",
                 num_layers=12,
                 num_stages=4):

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

            cond_ = dilated_conv(cond, "flow_{0}_cond_{1}".format(flow, i),
                                 filter_width,
                                 hidden_units,
                                 1,
                                 trainable=training)

            mid = int(hidden_units / 2)

            tanh = tf.tanh(lay_[:, :, :, :mid] + cond_[:, :, :, :mid])

            sig = tf.sigmoid(lay_[:, :, :, mid:] + cond_[:, :, :, mid:])

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

        location = params[:, :, :, :mid]

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
                 cond,
                 teacher,
                 flows,
                 filter_width,
                 hidden_units,
                 output_classes,
                 nb_mixtures,
                 training):

        h, w, in_ch, out_ch = inputs.shape

        h = int(h)
        w = int(w)
        in_ch = int(in_ch)
        out_ch = int(out_ch)

        flow = inputs

        location = tf.zeros(shape=inputs.shape, dtype=tf.float32)

        scale = tf.ones(shape=inputs.shape, dtype=tf.float32)

        for i in range(flows):

            # Double check for the number of layers in individual flows
            # Need to have different layers for different flows
            block = StudentWavenetComp(inputs=flow,
                                       cond=cond,
                                       flow=i,
                                       filter_width=filter_width,
                                       hidden_units=hidden_units,
                                       output_classes=2,
                                       training=training)

            flow = block.scale * flow + block.location

            location = location * block.scale + block.location
            scale = tf.abs(block.scale * scale)

        # logistic distribution
        log_dist = tf.contrib.distributions.Logistic(loc=location, scale=scale)

        samples = log_dist.sample([inputs.shape[0]])
        samples = tf.reshape(samples, [inputs.shape[0] * h, w, in_ch, out_ch])

        self.inputs = inputs
        self.cond = cond
        self.outputs = discretize(flow, 65535.) / 32676.5
        self.samples = samples
        self.location = location
        self.scale = scale
        self.teacher = teacher
        self.dist = log_dist.log_prob(inputs)
