from model import *

def generate():

    with tf.Session() as sess:

        # generate random noise input
        gen = np.random.logistic(0., 1., [32, 1, 7680, 1])

        # create IAF and load previously trained weights
        iaf = IAF(inputs=tf.placeholder(dtype=tf.float32, shape=gen.shape),
                  flows=4,
                  filter_width=3,
                  hidden_units=64,
                  training=False)

        load = tf.train.Saver()

        load.restore(sess, "/tmp/IAF.ckpt")

        print("Model restored")

        outputs = sess.run(iaf.outputs,
                           feed_dict={iaf.inputs: gen})

    return outputs


outputs = generate()

outputs = outputs / 32768. - 1.

outputs = outputs.reshape(outputs, (32 * 7680,))

for i in range(32):

    save_audio("/tmp/gen_file_{0}".format(i), outputs[32 * i: 32 * (i + 1)], 44100)
