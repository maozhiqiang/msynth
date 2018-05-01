from model import *
import sys


# number of samples in each block
block_size = 15

# number of sample blocks
nb_blocks = 1

# batch size
batch_size = 32

# Change the file_path to specify the path to your audio file
file = train_sine(block_size * nb_blocks)

# Compute the mu-law of the file to restrict the number of possible values. Here we use 16-bit mu-law
file = np.rint((mu_law(file, 65535)) * 32767.5) #/ 32767.5

shape = file.shape

# Reshape input to 4D
file = file.reshape([shape[0], 1, shape[1], 1])

# get probability distribution for the labels (soft probs). Used in KL loss during training
targets = np.full([batch_size, 1, block_size, 1], 1. / batch_size, dtype=np.float32)

# total number of batches
nb_batches = int(file.shape[0] / batch_size + 1) if file.shape[0] % batch_size != 0 else int(file.shape[0] / batch_size)
print(nb_batches)

def train_teacher():

    print("Training teacher")

    # train the TeacherWavenet
    with tf.Session() as sess:

        epoch = 0

        # current batch number
        batch_id = 0

        # current block number
        block_id = 0

        batch = get_batch(file, batch_id, block_id, batch_size, block_size)

        # Conditioning inputs. Start off with 0's.
        prev = np.zeros([batch_size, 1, block_size, 1], np.float32)

        h, w, in_ch, out_ch = file.shape

        # Create a single Wavenet structure
        train_net = TeacherWavenet(inputs=tf.placeholder(dtype=tf.float32, shape=batch.shape),
                                   cond=tf.placeholder(dtype=tf.float32, shape=batch.shape),
                                   targets=tf.placeholder(dtype=tf.float32, shape=targets.shape),
                                   filter_width=3,
                                   hidden_units=64,
                                   output_classes=2,
                                   training=True)

        # compute the cross entropy
        loss = tf.reduce_mean(tf.reduce_sum(-train_net.log_probs, 2))

        train_step = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # Saver to save the model after training
        saver = tf.train.Saver()

        while True:

            if batch_id == 0 and block_id == 0:
                print("Trainig epoch {0}".format(epoch))

            _, probs, m, s, error = sess.run([train_step, train_net.log_probs, train_net.location, train_net.scale, loss],
                                       feed_dict={train_net.inputs: batch, train_net.cond: prev, train_net.targets: targets})

            if np.isnan(error):
                print(batch, m, s, probs)

                break

            # change block_id for the next batch
            block_id = (block_id + 1) % nb_blocks

            # change batch_id for the next batch and check if at the end of inputs
            if block_id == 0:
                batch_id = (batch_id + 1) % nb_batches

                # reset to 0 since we generate beginning of sequence
                prev = np.zeros([batch_size, 1, block_size, 1], np.float32)

                if batch_id == 0:
                    epoch += 1

                    print("Error is: ", error)

            else:
                # use outputs as conditioning for other sample batches
                prev = get_batch(file, batch_id, block_id - 1, batch_size, block_size)

            if epoch >= 20:

                # Default location where the model will be saved. Can change it to anything
                save_path = saver.save(sess, "/tmp/ParallelWavenet.ckpt")

                print("Model saved in {0}".format(save_path))

                print(probs, m , s)

                break

    # reset graph
    tf.reset_default_graph()


def train_student():

    print("Training Student")

    # train the Student Wavenet
    with tf.Session() as sess:

        epoch = 0

        # current batch number
        batch_id = 0

        # current block number
        block_id = 0

        # noise inputs in batch
        gen = np.random.logistic(0., 1., [batch_size, 1, block_size, 1]) # increase the number of randm sample to get better entropy approx.

        h, w, in_ch, out_ch = gen.shape

        # conditioning input. Start off with 0's
        prev = np.zeros(gen.shape, np.float32)

        # create IAF and then train it
        iaf = IAF(inputs=tf.placeholder(dtype=tf.float32, shape=[batch_size, 1, block_size, 1]),
                  cond=tf.placeholder(dtype=tf.float32, shape=[batch_size, 1, block_size, 1]),
                  teacher=tf.placeholder(dtype=tf.float32, shape=[batch_size**2, 1, block_size, 1]),
                  flows=3,
                  filter_width=3,
                  hidden_units=64,
                  output_classes=2,
                  nb_mixtures=1,
                  training=True)

        # regenerate the Teacher Wavenet and optimize
        teacher = TeacherWavenet(inputs=iaf.samples,
                                 cond=tf.tile(iaf.cond, [batch_size, 1, 1, 1]),
                                 targets=None,
                                 filter_width=3,
                                 hidden_units=64,
                                 output_classes=2,
                                 training=False)

        restore = [var for var in tf.global_variables() if var not in tf.trainable_variables()]

        # use the KL divergence instead and feed it the teacher probs output
        loss = kl_div(teacher.log_probs, iaf.scale, batch_size)

        train_step = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)

        # Saver to load the weights of the teacher
        load = tf.train.Saver(restore)

        sess.run(tf.global_variables_initializer())

        load.restore(sess, "/tmp/ParallelWavenet.ckpt")


        # Saver to save the model after training
        saver = tf.train.Saver(tf.trainable_variables())

        print("Model restored")


        while True:

            if batch_id == 0 and block_id == 0:
                print("Training epoch {0}".format(epoch))

            # used to test when we vary the input. Does it yield better estimate of KL div
            gen_2 = np.random.logistic(0., 1., gen.shape)

            # run the model
            _, error, outputs, samples, m, s = sess.run([train_step, loss, iaf.outputs, iaf.samples, iaf.location, iaf.scale],
                                feed_dict={iaf.inputs: gen_2, iaf.cond: prev})

            # change block_id for the next batch
            block_id = (block_id + 1) % nb_blocks

            # change batch_id for the next batch and check if at the end of inputs
            if block_id == 0:

                # reset to 0 since we generate beginning of sequence
                prev = np.zeros(gen.shape, np.float32)

                epoch += 1

                print("Error is: ", error)

            else:
                # use outputs as conditioning for other sample batches
                prev = outputs

            if epoch >= 20:
                outputs, m, s = sess.run([iaf.outputs, iaf.location, iaf.scale], feed_dict={iaf.inputs:gen_2, iaf.cond:prev})

                save_path = saver.save(sess, "/tmp/IAF.ckpt")

                print("Model saved in {0}".format(save_path))

                print(outputs, np.mean(m, 0, keepdims=True), np.mean(s, 0, keepdims=True))

                break

    # reset graph
    tf.reset_default_graph()


def main():

    # train Teacher Wavenet
    if sys.argv[1] == "-t":
        train_teacher()

    # train Student Wavenet
    if sys.argv[1] == "-s":
        train_student()


if __name__ == "__main__":
    main()
