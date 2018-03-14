from model import *

# Change the file_path to specify the path to your audio file
file = train_sine(7680)

# Compute the mu-law of the file to restrict the number of possible values. Here we use 8-bit mu-law
file = np.floor((mu_law(file, 65535) + 1) * 32768)

shape = file.shape

# Reshape input to be used by dilated convolution layer
file = file.reshape([shape[0], 1, shape[1], 1])

# get probability distribution for the labels. Used in cross-entropy loss during training
targets = np.ones(shape=[shape[0], 1, shape[1], 1], dtype=np.float32)


def train_teacher():

    print("Training teacher")

    # train the TeacherWavenet
    with tf.Session() as sess:

        i = 0

        # Create a single Wavenet structure
        train_net = TeacherWavenet(inputs=tf.placeholder(dtype=tf.float32, shape=file.shape),
                                   targets=tf.placeholder(dtype=tf.float32, shape=file.shape),
                                   filter_width=3,
                                   hidden_units=256,
                                   output_classes=1,
                                   training=True)

        sess.run(tf.global_variables_initializer())

        # Saver to save the model after training
        saver = tf.train.Saver()

        while True:

            error, _ = sess.run([train_net.loss, train_net.train_step],
                                feed_dict={train_net.inputs: file, train_net.targets: targets})

            print("Error_{0} is: ".format(i), error)

            i += 1

            if error < 0.7:

                # Default location where the model will be saved. Can change it to anything
                save_path = saver.save(sess, "/tmp/ParallelWavenet.ckpt")

                print("Model saved in {0}".format(save_path))

                break

    # reset graph
    tf.reset_default_graph()


def train_student():

    print("Training Student")

    # train the Student Wavenet
    with tf.Session() as sess:

        i = 0

        # noise inputs in batch
        gen = np.random.logistic(0., 1., file.shape)

        # create IAF and then train it
        iaf = IAF(inputs=tf.placeholder(dtype=tf.float32, shape=file.shape),
                  flows=4,
                  filter_width=3,
                  hidden_units=64,
                  training=True)

        # Saver to load the weights of the teacher
        load = tf.train.Saver(iaf.restore)

        # Saver to save the model after training
        saver = tf.train.Saver()

        load.restore(sess, "/tmp/ParallelWavenet.ckpt")

        print("Model restored")

        sess.run(tf.global_variables_initializer())

        while True:

            error, _ = sess.run([iaf.loss, iaf.train_step],
                                feed_dict={iaf.inputs: gen})

            print("Error_{0} is: ".format(i), error)

            i += 1

            if error < 3000:
                save_path = saver.save(sess, "/tmp/IAF.ckpt")

                break

    # reset graph
    tf.reset_default_graph()

# train Teacher Wavenet
train_teacher()

# train Student Wavenet
train_student()
