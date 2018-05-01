import tensorflow as tf
import numpy as np
import pyo
import librosa


"""
Function that returns a causal convolution layer

Args:
    x: tensor of shape [batch_size, 1, time, in_channel]
    
    filter_width: width of the filter in timesteps
    
    out_channel: depth of the filter kernel. Number of layers of the kernel
    
    dilaton: dilation rate
    
    padding: "Valid" or "Same"

    
Returns:
    y: tensor of shape [batchsize, 1, out_width, out_channel]
"""


def dilated_conv(x,
                 name,
                 filter_width,
                 out_channel,
                 dilation=1,
                 padding="SAME",
                 trainable=True):

    # check if height of input is 1
    with tf.variable_scope(name):

        assert x.shape[1] == 1, x.shape[3] == 1
        assert name

        print(name, " X: ", x.shape)  # Used to test shapes. Remove after

        batch_size, _, time, in_channel = x.shape

        w = tf.get_variable(name="w",
                            shape=[1, filter_width, in_channel,  out_channel],
                            dtype=x.dtype,
                            initializer=tf.random_normal_initializer(stddev=0.01),
                            trainable=trainable)

        y = tf.nn.atrous_conv2d(value=x,
                                filters=w,
                                padding=padding,
                                rate=dilation)

        _, y_2, y_3, _ = y.shape

        b = tf.get_variable(name="b",
                            shape=[out_channel],
                            dtype=x.dtype,
                            initializer=tf.random_normal_initializer(stddev=0.0),
                            trainable=trainable)

        print(name, " Y: ", y.shape)  # Used to test shapes. Remove after

    return y + b


"""
Function that returns the mu-law of an input vector

Args:
    x: numpy array
    
    mu: 2 ** (number of bits)
    
    
Returns:
    y: tensor (or numpy array???) with the same shape as input with mu-law values
"""


def mu_law(x, mu):

    # normalize input between 1 and -1
    x_norm = x

    return np.sign(x_norm) * np.log(1 + np.abs(x_norm) * mu) / np.log(1 + mu)


"""
Function that returns the inverse mu-law

Args:
    x: numpy array

    mu: 2 ** (number of bits)
    
    
Returns:
    y: the inverse mu-law of x
"""


def inv_mu_law(x, mu):

    y = np.sign(x) * (1/mu) * ((1. + mu) ** np.abs(x) - 1.)

    return y


"""
Function that take a file and returns a numpy array

Args:
    file_path: absolute path to the file
    
    sample_rate: sampling rate of the audio file in hertz
    
    
Returns:
    output: tuple of numpy array and sampple rate
    
*** For now this function uses librosa --> implement using pyo ***
"""


def load_audio(file_path, sample_rate):

    return librosa.load(file_path, sample_rate)


"""
Function that takes a numpy array and returns an audio file

Args:
    save_path: absolute path to save the file
    
    x: input numpy array to save
    
    sample_rate: sampling rate of the audio file in hertz
    
Returns:
    output: audio file in .wav format

"""


def save_audio(save_path, x, sample_rate):

    return librosa.output.write_wav(save_path, x, sample_rate)


"""
Function that loads a batch of audio files in the wav format

Args:
    path: absolute path to directory containing sound files
    
    sample_rate: sampling rate of the audio files in hertz
    
Returns:
    output: batch of audio files in the numpy format
    
*** For now thus function uses librosa --> implement using pyo
"""


def load_batch(path, sample_rate):
    pass


"""
Function that takes vector and returns the labesl distribution to which it belongs

Args:
    

"""


def get_labels(x, bins):

    y = np.int32(np.floor(x * bins))
    y = y[0, 0, :, 0]

    out = np.zeros([1, 1, y.shape[0], bins])

    out[:, :, np.arange(y.shape[0]), y] = 1

    return out


"""
Function that returns an operator that computes the discrete version of a tensor (to nearest
16-bit value)

Args:
    x: input tensor
    
    nb_bins: 2^number_bits - 1
     
Returns:
    discrete tensor: operator that can compute the discrete version of x
"""


def discretize(x, nb_bins):

    discrete_x = tf.where(x <= -1., tf.fill(x.shape, -1.), x)

    discrete_x = tf.where(discrete_x >= 1., tf.fill(x.shape, 1.), discrete_x)

    discrete_x = tf.round(discrete_x * (nb_bins / 2)) / (nb_bins / 2)

    return discrete_x


"""
Function that is used to return the KL divergence between the probabilites of the student
and the teacher

Args:
    p_t: log probabilities output by the teacher network
    
    scale: scale of the IAF distribution for each sample
    
    num_samples: number of samples for each x_t to calculate cross entropy
    
Returns:
    loss: the KL divergence between the two probability distributions 
"""


def kl_div(p_t, scale, num_samples):

    p_t_ = tf.reshape(p_t, [num_samples, int(int(p_t.shape[0]) / num_samples), 1, p_t.shape[2], 1])

    # cross entropy term
    loss = tf.reduce_mean(-p_t_, 0)

    # expectation over p_s(x_<t) and then sum over t=1,...,T
    loss = tf.reduce_sum(tf.reduce_mean(loss, axis=0))

    loss -= tf.reduce_mean(tf.reduce_sum(tf.log(scale), 2)) + 2. * int(scale.shape[2])

    return loss


"""
Function that returns the power loss of a signal. Essentially finds the spectograms
of the generated audio snippet and the input and subtracts them. It then computes the
magnitude-squared of the result to get the power loss

Args:
    gen: generated audio input
    
    x: existing input
    
Returns:
    loss: power loss given the inputs
"""


def power_loss(gen, x):
    pass


"""
Function that generates a bunch of sine waves at different frequencies that will be used
to train the model as an example.

Args:
    length: duration of the audio stream in samples

Returns:
    list: numpy array of sine waves as arrays of numbers that will be used by the program to train
"""


def train_sine(length):

    freq_list = [32 + (i + 1) for i in range(1024)]

    sine_list = np.array([[np.sin(2. * np.pi * i * j / 44100.) for i in range(length)] for j in freq_list])

    return sine_list


def get_batch(x, batch, block, batch_size, block_size):

    return x[batch * batch_size:(batch + 1) * batch_size, :, block * block_size:(block + 1) * block_size, :]
