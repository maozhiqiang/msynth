from model import *

# Change the file_path to specify the path to your audio file
file, _ = load_audio("/Users/williamst-arnaud/U de M/MUS3325/msynth/03 Plimsoll Punks.wav", sample_rate=44100)

# Compute the mu-law of the file to restrict the number of possible values. Here we use 8-bit mu-law
file = mu_law(file, 255)[:1024]

shape = file.shape

#Reshape input to be used by dilated convolution layer
file = file.reshape([1, 1, shape[0], 1])

# get probability distribution for the labels. Used in cross-entropy loss during training
#targets = get_labels(file, 256) this is for categorical
# targets = file * 128 + 128
targets = np.ones(shape=[1,1,1024,1], dtype=np.float32)


# Use these instead for the feed_dict to test behaviour
temp = np.random.rand(1, 1, 1024, 1)
temp1 = np.random.rand(1, 1, 1024, 256)
temp1 = temp1 / np.sum(temp1, axis=2, keepdims=True)



with tf.Session() as sess:

    i = 0

    # Create a single Wavenet structure
    example = TeacherWavenet(filter_width=3,
                      hidden_units=64,
                      output_classes=1,
                      training=True)

    sess.run(tf.global_variables_initializer())


    while True:
        error, _ = sess.run([example.loss, example.train_step], feed_dict={example.inputs: file, example.targets: targets})
        print("Error_{0} is: ".format(i), error)
        i+=1

        if error < 1:
            break
