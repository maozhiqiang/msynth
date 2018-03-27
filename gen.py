from model import *

sess = tf.Session()

def restore(sess, shape):

    # create IAF and load previously trained weights
    iaf = IAF(inputs=tf.placeholder(dtype=tf.float32, shape=shape),
              flows=1,
              filter_width=3,
              hidden_units=2,
              output_classes=2,
              training=False)

    load = tf.train.Saver()

    load.restore(sess, "/tmp/IAF.ckpt")

    print("Model restored")

    return iaf


def generate(sess, iaf):

    # generate random noise input
    gen = np.random.logistic(0., 1., [32, 1, 10, 1])

    outputs = sess.run(iaf.outputs,
                       feed_dict={iaf.inputs: gen})

    return outputs


iaf = restore(sess, [32, 1, 10, 1])

outputs = generate(sess, iaf)

for i in range(9999):

    outputs = np.concatenate([outputs, generate(sess, iaf)], axis=2)

    # outputs = outputs / 32767.5 - 1.

file = outputs.reshape([outputs.shape[0], outputs.shape[2]])

for i in range(2):

    save_audio("/tmp/gen_file_{0}.wav".format(i), file[i], 44100)
