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

        print(name, " X: ", x.shape)  # Used to test shapes. Remove after

        batch_size, _, time, in_channel = x.shape

        init1 = tf.constant_initializer(np.zeros(shape=[batch_size, filter_width, in_channel, out_channel],
                                                 dtype=np.float32))

        w = tf.get_variable(name="w",
                            shape=[batch_size, filter_width, in_channel,  out_channel],
                            dtype=x.dtype,
                            initializer=tf.random_normal_initializer(stddev=0.1),
                            trainable=trainable)

        y = tf.nn.atrous_conv2d(value=x,
                                filters=w,
                                padding=padding,
                                rate=dilation)

        _, y_2, y_3, _ = y.shape

        init2 = tf.constant_initializer(np.zeros(shape=[batch_size, y_2, y_3, out_channel],
                                                dtype=np.float32))

        b = tf.get_variable(name="b",
                            shape=[batch_size, y_2, y_3, out_channel],
                            dtype=x.dtype,
                            initializer=tf.random_normal_initializer(stddev=0.1),
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
Function that takes a numpy array and returns an audio file

Args:
    save_path: absolute path to save the file
    
    input: input numpy array to save
    
    sample_rate: sampling rate of the audio file in hertz
    
Returns:
    output: audio file in .wav format

"""


def save_audio(save_path, input, sample_rate):

    return librosa.output.write_wav(save_path , input, sample_rate)


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


def disc_log_mixt(x, params, maximum, minimum):

    m = params[0]
    s = params[1]

    upper = tf.sigmoid((x + 0.5 - m) / s)

    lower = tf.sigmoid((x - 0.5 - m) / s)

    func = tf.where(x >= maximum, tf.ones(dtype=tf.float32, shape=x.shape), upper)

    func = tf.where(func <= minimum, tf.zeros(dtype=tf.float32, shape=x.shape), lower)

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

    t_probs_0 = tf.where(t_probs <= 0., t_probs + 0.01, t_probs)

    entropy_term = tf.reduce_mean(tf.reduce_sum(tf.log(scale), 2)) + 2 * int(z.shape[2])

    cross_entropy_term = - tf.reduce_sum(tf.reduce_mean((s_probs * tf.log(t_probs_0)), 0))

    return cross_entropy_term - entropy_term


"""
Function that returns the power loss of a signal. Essentially finds the spectograms
of the generated audio snippet and the input and subtracts them. It then computes the
magnitude-squared of the result to get the power loss

Args:
    gen: generated audio input
    
    input: existing input
    
Returns:
    loss: power loss given the inputs
"""


def power_loss(gen, input):
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

    freq_list = [32 * i for i in range(1,33)]

    sine_list = np.array([[np.sin(2. * np.pi * i * j / 44100.) for i in range(length)] for j in freq_list])

    return sine_list
