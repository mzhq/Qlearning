import tensorflow as tf
import gym

class CNN:
    '''
        CNN
        conv4
    '''
    def __init__(self, env):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=1e-3)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(1e-3, shape=shape)
            return tf.Variable(initial)

        def conv2d(input_tensor, W):
            return tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding='SAME')

        def use_relu(conv, conv_biases):
            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

        self.input_length = 8 * 8 * 3
        self.input_s = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.input_1 = tf.reshape(self.input_s, shape=[1, 8, 8, 3])
        # layer1-conv64
        self.W_conv_1 = weight_variable([3, 3, 3, 64])
        self.b_1 = bias_variable([64])
        self.conv_1 = conv2d(self.input_1, self.W_conv_1)
        self.out_1 = use_relu(self.conv_1, self.b_1)

        # layer2-conv64
        self.W_conv_2 = weight_variable([3, 3, 64, 64])
        self.b_2 = bias_variable([64])
        self.conv_2 = conv2d(self.out_1, self.W_conv_2)
        self.out_2 = use_relu(self.conv_2, self.b_2)

        # layer3-conv128
        self.W_conv_3 = weight_variable([3, 3, 64, 128])
        self.b_3 = bias_variable([128])
        self.conv_3 = conv2d(self.out_2, self.W_conv_3)
        self.out_3 = use_relu(self.conv_3, self.b_3)

        # layer4-conv128
        self.W_conv_4 = weight_variable([3, 3, 128, 128])
        self.b_4 = bias_variable([128])
        self.conv_4 = conv2d(self.out_3, self.W_conv_4)
        self.out_4 = use_relu(self.conv_4, self.b_4)

        # layer5-fc128
        self.out_4_flat = tf.reshape(self.out_4, [-1, 8 * 8 * 128])
        self.W_5 = weight_variable([8 * 8 * 128, 128])
        self.b_5 = bias_variable([128])
        self.out_5 = tf.nn.relu(tf.matmul(self.out_4_flat, self.W_5) + self.b_5)

        # layer6-fc60
        self.W_6 = weight_variable([128, env.action_space.n])
        self.b_6 = bias_variable([env.action_space.n])
        self.Q = tf.nn.relu(tf.matmul(self.out_5, self.W_6) + self.b_6)

        self.Q_target = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q))
        self.update = tf.train.GradientDescentOptimizer(1e-2).minimize(self.loss)
        self.init = tf.global_variables_initializer()

class Freezing_CNN:
    def __init__(self, env):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=1e-2)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(1e-2, shape=shape)
            return tf.Variable(initial)

        def conv2d(input_tensor, W):
            return tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding='SAME')

        def use_relu(conv, conv_biases):
            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

        self.input_length = 8 * 8 * 3
        self.input_s = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.input_1 = tf.reshape(self.input_s, shape=[1, 8, 8, 3])

        with tf.variable_scope('main_model'):
            # layer1-conv64
            W_conv_1 = weight_variable([3, 3, 3, 64])
            b_1 = bias_variable([64])
            conv_1 = conv2d(self.input_1, W_conv_1)
            out_1 = use_relu(conv_1, b_1)

            # layer2-conv64
            W_conv_2 = weight_variable([3, 3, 64, 64])
            b_2 = bias_variable([64])
            conv_2 = conv2d(out_1, W_conv_2)
            out_2 = use_relu(conv_2, b_2)

            # layer3-conv128
            W_conv_3 = weight_variable([3, 3, 64, 128])
            b_3 = bias_variable([128])
            conv_3 = conv2d(out_2, W_conv_3)
            out_3 = use_relu(conv_3, b_3)

            # layer4-conv128
            W_conv_4 = weight_variable([3, 3, 128, 128])
            b_4 = bias_variable([128])
            conv_4 = conv2d(out_3, W_conv_4)
            out_4 = use_relu(conv_4, b_4)

            # layer5-fc128
            out_4_flat = tf.reshape(out_4, [-1, 8 * 8 * 128])
            W_5 = weight_variable([8 * 8 * 128, 128])
            b_5 = bias_variable([128])
            out_5 = tf.nn.relu(tf.matmul(out_4_flat, W_5) + b_5)

            # layer6-fc60
            W_6 = weight_variable([128, env.action_space.n])
            b_6 = bias_variable([env.action_space.n])
            self.Q = tf.nn.relu(tf.matmul(out_5, W_6) + b_6)


        self.Q_target = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q))
        self.update = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)
        self.init = tf.global_variables_initializer()


        with tf.variable_scope('freezing_model'):
            # layer1-conv64
            W_conv_1 = weight_variable([3, 3, 3, 64])
            b_1 = bias_variable([64])
            conv_1 = conv2d(self.input_1, W_conv_1)
            out_1 = use_relu(conv_1, b_1)

            # layer2-conv64
            W_conv_2 = weight_variable([3, 3, 64, 64])
            b_2 = bias_variable([64])
            conv_2 = conv2d(out_1, W_conv_2)
            out_2 = use_relu(conv_2, b_2)

            # layer3-conv128
            W_conv_3 = weight_variable([3, 3, 64, 128])
            b_3 = bias_variable([128])
            conv_3 = conv2d(out_2, W_conv_3)
            out_3 = use_relu(conv_3, b_3)

            # layer4-conv128
            W_conv_4 = weight_variable([3, 3, 128, 128])
            b_4 = bias_variable([128])
            conv_4 = conv2d(out_3, W_conv_4)
            out_4 = use_relu(conv_4, b_4)

            # layer5-fc128
            out_4_flat = tf.reshape(out_4, [-1, 8 * 8 * 128])
            W_5 = weight_variable([8 * 8 * 128, 128])
            b_5 = bias_variable([128])
            out_5 = tf.nn.relu(tf.matmul(out_4_flat, W_5) + b_5)

            # layer6-fc60
            W_6 = weight_variable([128, env.action_space.n])
            b_6 = bias_variable([env.action_space.n])
            self.freezing_Q = tf.nn.relu(tf.matmul(out_5, W_6) + b_6)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='freezing_model')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_model')
        self.replace_model_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

class Simple_model:
    '''
        a simple neural network
        use flatten input
    '''
    def __init__(self, env):
        '''
        self.input_length = env.board_size ** 2 * 3
        self.input_s = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform(shape=[self.input_length, env.action_space.n], minval=-1e-4, maxval=1e-4), name = 'w')
        # self.Q = tf.matmul(self.input_s, self.W)
        self.b1 = tf.Variable(tf.zeros([1, env.action_space.n]) + 1e-4)
        self.Q = tf.nn.relu(tf.matmul(self.input_s, self.W) + self.b1)
    '''
        self.input_length = env.board_size ** 2 * 3
        self.input_s = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.W1 = tf.Variable(tf.random_uniform(shape=[self.input_length, 10], minval=-1e-4, maxval=1e-4))
        # self.Q = tf.matmul(self.input_s, self.W)
        self.b1 = tf.Variable(tf.zeros([1, 10]) + 1e-4)
        self.out_1 = tf.nn.relu(tf.matmul(self.input_s, self.W1) + self.b1)

        self.W2 = tf.Variable(tf.random_uniform(shape=[10, env.action_space.n], minval=-1e-4, maxval=1e-4))
        self.b2 = tf.Variable(tf.zeros([1, env.action_space.n]) + 1e-4)
        self.Q = tf.nn.relu(tf.matmul(self.out_1, self.W2) + self.b2)
        # self.predict_action = tf.argmax(self.Q, 1)

        self.Q_target = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q))
        self.update = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)
        self.init = tf.global_variables_initializer()