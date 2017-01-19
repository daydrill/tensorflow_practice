import numpy as np
import tensorflow as tf
from chad_practice.QRNN.tf_qrnn import QRNN


class TestQRNNForward():

    def test_qrnn_linear_forward(self):
        batch_size = 100
        sentence_length = 5
        word_size = 10
        size = 5
        data = self.create_test_data(batch_size, sentence_length, word_size)
        print(data)

        with tf.Graph().as_default() as q_linear:
            qrnn = QRNN(in_size=word_size, size=size, conv_size=1)
            X = tf.placeholder(tf.float32, [batch_size, sentence_length, word_size])
            forward_graph = qrnn.forward(X)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                hidden = sess.run(forward_graph, feed_dict={X: data})
                # print(hidden)
                # self.assertEqual((batch_size, size), hidden.shape)

    def test_qrnn_with_previous(self):
        batch_size = 100
        sentence_length = 5
        word_size = 10
        size = 5
        data = self.create_test_data(batch_size, sentence_length, word_size)

        with tf.Graph().as_default() as q_with_previous:
            qrnn = QRNN(in_size=word_size, size=size, conv_size=2)
            X = tf.placeholder(tf.float32, [batch_size, sentence_length, word_size])
            forward_graph = qrnn.forward(X)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                hidden = sess.run(forward_graph, feed_dict={X: data})
                self.assertEqual((batch_size, size), hidden.shape)

    def test_qrnn_convolution(self):
        batch_size = 100
        sentence_length = 5
        word_size = 10
        size = 5
        data = self.create_test_data(batch_size, sentence_length, word_size)

        with tf.Graph().as_default() as q_conv:
            qrnn = QRNN(in_size=word_size, size=size, conv_size=3)
            X = tf.placeholder(tf.float32, [batch_size, sentence_length, word_size])
            forward_graph = qrnn.forward(X)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                hidden = sess.run(forward_graph, feed_dict={X: data})
                self.assertEqual((batch_size, size), hidden.shape)

    def create_test_data(self, batch_size, sentence_length, word_size):
        batch = []
        for b in range(batch_size):
            sentence = np.random.rand(sentence_length, word_size)
            batch.append(sentence)
        return np.array(batch)


if __name__ == "__main__":
    tqrnn = TestQRNNForward()
    # tqrnn.create_test_data()

    tqrnn.test_qrnn_linear_forward()