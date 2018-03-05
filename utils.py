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

        assert x.shape[1] == 1
        assert name

        print("X: ", x.shape)  # Used to test shapes. Remove after

        batch_size, _, time, in_channel = x.shape

        w = tf.get_variable(name="w",
                            shape=[1, filter_width, in_channel,  out_channel],
                            dtype=x.dtype,
                            initializer=tf.random_normal_initializer(),
                            trainable=trainable)

        y = tf.nn.atrous_conv2d(value=x,
                                filters=w,
                                padding=padding,
                                rate=dilation)

        _, y_2, y_3, _ = y.shape

        b = tf.get_variable(name="b",
                            shape=[1, y_2, y_3, out_channel],
                            dtype=x.dtype,
                            initializer=tf.random_normal_initializer(),
                            trainable=trainable)

        print("Y: ", y.shape)  # Used to test shapes. Remove after

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

    y = np.sign(x) * (np.sign(x) * np.log(1 + mu) * np.exp(x) - 1)

    return y * mu / 2


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
Function that returns an operator that computes the discrete log mixture to be used instead 
of the categorial distribution as it is more tractable

Args:
    x: input tensor
    
    shape: shape of input tensor that will be fed at run time
    
    params: parameter tuple (location, scale) to be used in mixture model
    
    max: max possible value of discrete input x
    
    min: min possbile value of discrete input x
     
Returns:
    op: operator that can compute the discretized prob of an input x
    
*** Change the +/- 0.5 tu upper bound / lower bound, add # bits???Remove shape?????***
"""


def disc_log_mixt(x, params, max, min):

    func = 0

    upper = tf.where(x >= max, x + np.inf, x + 0.5)

    lower = tf.where(x <= min, x - np.inf, x - 0.5)

    m = params[0]
    s = params[1]

    func = tf.sigmoid((upper - m) / s) - tf.sigmoid((lower - m) / s)

    print(func)

    return func


"""
Function that is used to return the KL divergence between the probabilites of the student
and the teacher

Args:
    z: tensor (noise) that is fed to the IAF to generate output

    s_probs: probabilities returned by a the IAF for a given input
    
    t_probs: probabilities returned by the TeacherWavenet for the same input
    
    location: location of the IAF distribution for each sample
    
    scale: scale of the IAF distribution for each sample
    
Returns:
    loss: the KL divergence between the two probability distributions 
"""


def kull_leib(z, s_probs, t_probs, location, scale):

    entropy_term = tf.reduce_mean(tf.reduce_sum(tf.log(scale), 2)) + 2 * int(z.shape[2])

    cross_entropy_term = - tf.reduce_sum(tf.reduce_mean((s_probs * tf.log(t_probs)), 0))

    return cross_entropy_term - entropy_term
