from model import *

sess = tf.Session()


def restore(session, shape):

    # create IAF and load previously trained weights
    flow = IAF(inputs=tf.placeholder(dtype=tf.float32, shape=shape),
               cond=tf.placeholder(dtype=tf.float32, shape=shape),
               teacher=None,
               flows=1,
               filter_width=3,
               hidden_units=64,
               output_classes=2,
               nb_mixtures=2,
               training=False)

    load = tf.train.Saver()

    load.restore(session, "/tmp/IAF.ckpt")

    print("Model restored")

    return flow


def generate(session, flow, cond):

    # generate random noise input
    gen = np.random.logistic(0., 1., [32, 1, 10, 1])

    gen_audio = session.run(flow.outputs,
                            feed_dict={flow.inputs: gen, flow.cond: cond})

    gen_audio = inv_mu_law(gen_audio, 65535.)

    return gen_audio


def main():

    iaf = restore(sess, [32, 1, 10, 1])

    gen_block = generate(sess, iaf, np.zeros([32, 1, 10, 1], np.float32))

    print(gen_block)

    outputs = gen_block

    for i in range(1):

        gen_block = generate(sess, iaf, gen_block)

        outputs = np.concatenate([outputs, gen_block], axis=2)

    print("Generated audio files")

    file = outputs.reshape([outputs.shape[0], outputs.shape[2]])

    for i in range(2):

        save_audio("/tmp/gen_file_{0}.wav".format(i), file[i], 44100)


if __name__ == "__main__":
    main()
