from model import *

# Change the file_path to specify the path to your audio file
file, _ = load_audio("/Users/williamst-arnaud/U de M/MUS3325/msynth/03 Plimsoll Punks.wav", sample_rate=44100)

# Compute the mu-law of the file to restrict the number of possible values. Here we use 8-bit mu-law
file = mu_law(file, 255)[:1024]

shape = file.shape

# Reshape input to be used by dilated convolution layer
file = file.reshape([1, 1, shape[0], 1])

# get probability distribution for the labels. Used in cross-entropy loss during training
targets = np.ones(shape=[1, 1, 1024, 1], dtype=np.float32)

# train the TeacherWavenet
with tf.Session() as train_sess:

    i = 0

    # Create a single Wavenet structure
    train_net = TeacherWavenet(inputs=tf.placeholder(dtype=tf.float32, shape=[1, 1, 1024, 1]),
                               targets=tf.placeholder(dtype=tf.float32, shape=[1, 1, 1024, 1]),
                               filter_width=3,
                               hidden_units=64,
                               output_classes=1,
                               training=True)

    train_sess.run(tf.global_variables_initializer())

    # Saver to save the model after training
    saver = tf.train.Saver()

    while True:
        error, _ = train_sess.run([train_net.loss, train_net.train_step],
                                  feed_dict={train_net.inputs: file, train_net.targets: targets})

        print("Error_{0} is: ".format(i), error)

        i += 1

        if error < 0.1:

            # Default location where the model will be saved. Can change it to anything
            save_path = saver.save(train_sess, "/tmp/ParallelWavenet.ckpt")

            print("Model saved in {0}".format(save_path))

            break

# reset graph
tf.reset_default_graph()

""" Output predictions for audio files using trained TeacherWavenet
Using the same file as used in training, therefore the probabilities will be high
Eventually, we'll be using the generated audio files from the student Wavenet
"""

with tf.Session() as pred_sess:

    i = 0

    # create IAF and then train it
    iaf = IAF(flows=4,
              filter_width=3,
              hidden_units=64,
              training=True)

    # Saver to save the model after training
    saver = tf.train.Saver(iaf.restore)

    saver.restore(pred_sess, "/tmp/ParallelWavenet.ckpt")

    print("Model restored")

    pred_sess.run(tf.global_variables_initializer())

    while True:

        error, _ = pred_sess.run([iaf.loss, iaf.train_step],
                                      feed_dict={iaf.inputs: file})

        print("The gen error_{0} is {1}".format(i, error))

        i += 1

        if error < 0.1:

            save_path = saver.save(pred_sess, "/tmp/IAF.ckpt")

            break
