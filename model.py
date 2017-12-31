import tensorflow as tf
from layers import depthwise_separable_conv2d, conv2d, avg_pool_2d, dense, flatten, dropout
import os
from utils import load_obj, save_obj
import numpy as np


class MobileNet:
    """
    MobileNet Class
    """

    def __init__(self,
                 args):

        # init parameters and input
        self.X = None
        self.y = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.regularization_loss = None
        self.cross_entropy_loss = None
        self.train_op = None
        self.accuracy = None
        self.y_out_argmax = None
        self.summaries_merged = None
        self.args = args
        self.mean_img = None

        self.pretrained_path = os.path.realpath(self.args.pretrained_path)

        # All layers
        self.conv1_1 = None

        self.conv2_1 = None
        self.conv2_2 = None

        self.conv3_1 = None
        self.conv3_2 = None

        self.conv4_1 = None
        self.conv4_2 = None

        self.conv5_1 = None
        self.conv5_2 = None
        self.conv5_3 = None
        self.conv5_4 = None
        self.conv5_5 = None
        self.conv5_6 = None

        self.conv6_1 = None
        self.flattened = None

        self.__build()

    def __init_input(self):
        with tf.variable_scope('input'):
            # Input images
            self.X = tf.placeholder(tf.float32,
                                    [self.args.batch_size, self.args.img_height, self.args.img_width,
                                     self.args.num_channels])
            # Classification supervision, it's an argmax. Feel free to change it to one-hot,
            # but don't forget to change the loss from sparse as well
            self.y = tf.placeholder(tf.int32, [self.args.batch_size])
            # is_training is for batch normalization and dropout, if they exist
            self.is_training = tf.placeholder(tf.bool)

    def __init_mean(self):
        # Preparing the mean image.
        img_mean = np.ones((1, 224, 224, 3))
        img_mean[:, :, :, 0] *= 103.939
        img_mean[:, :, :, 1] *= 116.779
        img_mean[:, :, :, 2] *= 123.68
        self.mean_img = tf.constant(img_mean, dtype=tf.float32)

    def __build(self):
        self.__init_mean()
        self.__init_input()
        self.__init_network()
        self.__init_output()

    def __init_network(self):
        with tf.variable_scope('mobilenet_encoder'):
            # Preprocessing as done in the paper
            with tf.name_scope('pre_processing'):
                preprocessed_input = (self.X - self.mean_img) / 255.0

            # Model is here!
            self.conv1_1 = conv2d('conv_1', preprocessed_input, num_filters=int(round(32 * self.args.width_multiplier)),
                                  kernel_size=(3, 3),
                                  padding='SAME', stride=(2, 2), activation=tf.nn.relu,
                                  batchnorm_enabled=self.args.batchnorm_enabled,
                                  is_training=self.is_training, l2_strength=self.args.l2_strength, bias=self.args.bias)
            ############################################################################################
            self.conv2_1 = depthwise_separable_conv2d('conv_ds_2', self.conv1_1,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=64, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv2_2 = depthwise_separable_conv2d('conv_ds_3', self.conv2_1,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            ############################################################################################
            self.conv3_1 = depthwise_separable_conv2d('conv_ds_4', self.conv2_2,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv3_2 = depthwise_separable_conv2d('conv_ds_5', self.conv3_1,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            ############################################################################################
            self.conv4_1 = depthwise_separable_conv2d('conv_ds_6', self.conv3_2,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv4_2 = depthwise_separable_conv2d('conv_ds_7', self.conv4_1,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            ############################################################################################
            self.conv5_1 = depthwise_separable_conv2d('conv_ds_8', self.conv4_2,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv5_2 = depthwise_separable_conv2d('conv_ds_9', self.conv5_1,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv5_3 = depthwise_separable_conv2d('conv_ds_10', self.conv5_2,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv5_4 = depthwise_separable_conv2d('conv_ds_11', self.conv5_3,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv5_5 = depthwise_separable_conv2d('conv_ds_12', self.conv5_4,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            self.conv5_6 = depthwise_separable_conv2d('conv_ds_13', self.conv5_5,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            ############################################################################################
            self.conv6_1 = depthwise_separable_conv2d('conv_ds_14', self.conv5_6,
                                                      width_multiplier=self.args.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=self.args.batchnorm_enabled,
                                                      activation=tf.nn.relu,
                                                      is_training=self.is_training,
                                                      l2_strength=self.args.l2_strength,
                                                      biases=(self.args.bias, self.args.bias))
            ############################################################################################
            self.avg_pool = avg_pool_2d(self.conv6_1, size=(7, 7), stride=(1, 1))
            self.dropped = dropout(self.avg_pool, self.args.dropout_keep_prob, self.is_training)
            self.logits = flatten(conv2d('fc', self.dropped, kernel_size=(1, 1), num_filters=self.args.num_classes,
                                 l2_strength=self.args.l2_strength,
                                 bias=self.args.bias))

    def __init_output(self):
        with tf.variable_scope('output'):
            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='loss'))
            self.loss = self.regularization_loss + self.cross_entropy_loss

            # Important for Batch Normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)
            self.y_out_argmax = tf.argmax(tf.nn.softmax(self.logits), axis=-1, output_type=tf.int32)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))

        # Summaries needed for TensorBoard
        with tf.name_scope('train-summary-per-iteration'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.accuracy)
            self.summaries_merged = tf.summary.merge_all()

    def __restore(self, file_name, sess):
        try:
            print("Loading ImageNet pretrained weights...")
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mobilenet_encoder")
            dict = load_obj(file_name)
            run_list = []
            for variable in variables:
                for key, value in dict.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))

            sess.run(run_list)
            print("ImageNet Pretrained Weights Loaded Initially\n\n")
        except KeyboardInterrupt:
            print("No pretrained ImageNet weights exist. Skipping...\n\n")

    def load_pretrained_weights(self, sess):
        # self.__convert_graph_names(os.path.realpath('pretrained_weights/mobilenet_v1_vanilla.pkl'))
        self.__restore(self.pretrained_path, sess)

    def __convert_graph_names(self, path):
        """
        This function is to convert from the mobilenet original model pretrained weights structure to our
        model pretrained weights structure.
        :param path: (string) path to the original pretrained weights .pkl file
        :return: None
        """
        dict = load_obj(path)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mobilenet_encoder')
        dict_output = {}

        for key, value in dict.items():
            for variable in variables:
                for i in range(len(dict)):
                    for j in range(len(variables)):
                        if ((key.find("Conv2d_" + str(i) + "_") != -1 and variable.name.find(
                                        "conv_ds_" + str(j) + "/") != -1) and i + 1 == j):
                            if key.find("depthwise") != -1 and variable.name.find(
                                    "depthwise") != -1 and (key.find("gamma") != -1 and variable.name.find(
                                "gamma") != -1 or key.find("beta") != -1 and variable.name.find(
                                "beta") != -1) or key.find("pointwise") != -1 and variable.name.find(
                                "pointwise") != -1 and (key.find("gamma") != -1 and variable.name.find(
                                "gamma") != -1 or key.find("beta") != -1 and variable.name.find(
                                "beta") != -1) or key.find("pointwise/weights") != -1 and variable.name.find(
                                "pointwise/weights") != -1 or key.find(
                                "depthwise_weights") != -1 and variable.name.find(
                                "depthwise/weights") != -1 or key.find("pointwise/biases") != -1 and variable.name.find(
                                "pointwise/biases") != -1 or key.find("depthwise/biases") != -1 and variable.name.find(
                                "depthwise/biases") != -1 or key.find("1x1/weights") != -1 and variable.name.find(
                                "1x1/weights") != -1 or key.find("1x1/biases") != -1 and variable.name.find(
                                "1x1/biases") != -1:
                                dict_output[variable.name] = value
                        elif key.find(
                                "Conv2d_0/") != -1 and variable.name.find("conv_1/") != -1:
                            if key.find("weights") != -1 and variable.name.find("weights") != -1 or key.find(
                                    "biases") != -1 and variable.name.find(
                                "biases") != -1 or key.find("beta") != -1 and variable.name.find(
                                "beta") != -1 or key.find("gamma") != -1 and variable.name.find(
                                "gamma") != -1:
                                dict_output[variable.name] = value
                        elif key.find("Logits") != -1 and variable.name.find("fc") != -1:
                            if key.find("weights") != -1 and variable.name.find("weights") != -1 or key.find(
                                    "biases") != -1 and variable.name.find("biases") != -1:
                                dict_output[variable.name] = value

        save_obj(dict_output, self.pretrained_path)
        print("Pretrained weights converted to the new structure. The filename is mobilenet_v1.pkl.")
