from model import *

# Change the file_path to specify the path to your audio file
file, _ = load_audio("/Users/williamst-arnaud/U de M/MUS3325/msynth/03 Plimsoll Punks.wav", sample_rate=44100)

# Compute the mu-law of the file to restrict the number of possible values. Here we use 8-bit mu-law
file = mu_law(file, 8)[:1024]

shape = file.shape

#Reshape input to be used by dilated convolution layer
file = file.reshape([1, 1, shape[0], 1])

#Ignore this part for now. Problem with datatypes not recognized by Placeholder ???????
targets = np.floor(file * 256)
targets = tf.one_hot(targets, depth=256)


targets = tf.reshape(targets, shape=[1, 1, 1024, 256])


# Use these instead for the feed_dict to test behaviour
temp = np.random.rand(1, 1, 1024, 1)
temp1 = np.random.rand(1, 1, 1024, 256)



with tf.Session() as sess:

    # Create a single Wavenet structure
    example = Wavenet(filter_width=3,
                      hidden_units=64,
                      output_classes=256)

    sess.run(tf.global_variables_initializer())

    a = sess.run(example.loss, feed_dict={example.inputs: file, example.targets: temp1})
    print("Error is: ", a)

"""
Loop for training


    while True:
        error = sess.run(example.train())

        print(error)

        if error < 0.5:
            break
"""
