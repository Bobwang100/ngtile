import os
import time
import shutil
from datetime import timedelta
from models import vgg_train
import numpy as np
import tensorflow as tf
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNet:
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob, num_inter_threads, num_intra_threads,
                 weight_decay, nesterov_momentum, model_type,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape  # image shape, like [224,224,3]
        self.n_classes = data_provider.n_classes
        self.depth = depth
        self.growth_rate = growth_rate
        self.num_inter_threads = num_inter_threads
        self.num_intra_threads = num_intra_threads
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()

        # Specify the CPU inter and Intra threads used by MKL
        config.intra_op_parallelism_threads = self.num_intra_threads
        config.inter_op_parallelism_threads = self.num_inter_threads

        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path, graph=self.sess.graph)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        # try:
        #     logs_path = self._logs_path
        # except AttributeError:
        #     logs_path = 'logs/%s' % self.model_identifier
        #     if self.renew_logs:
        #         shutil.rmtree(logs_path, ignore_errors=True)
        #     os.makedirs(logs_path, exist_ok=True)
        #     self._logs_path = logs_path
        # return logs_path
        return './models/logs_dense/'

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}".format(
            self.model_type, self.growth_rate, self.depth)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save1_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            self.logits = self.transition_layer_to_classes(output)
        prediction = tf.nn.softmax(self.logits)

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(
            cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train_all_epochs(self, train_params):
        self.saver.restore(self.sess, './model_dense_0110/transfer_learn_80000')
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        img_names_q, img_names_uq, img_num = vgg_train.load_all_img_names()
        aa, bb = [1, 0], [0, 1]
        len_qual, len_unqual = len(img_names_q), len(img_names_uq)
        qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
        unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)
        all_img_names = np.array(img_names_q + img_names_uq)
        ys = np.concatenate((qual_y, unqual_y), axis=0)

        for epoch in range(1, n_epochs + 1):
            # print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)
            # print("Training...")
            b_idx = np.random.randint(0, img_num, 4)
            xs = vgg_train.load_img_by_name(all_img_names[b_idx])

            feed_dict = {
                self.images: xs,
                self.labels: ys[b_idx],
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            if epoch % 500 == 0 or loss > 0.069:
                print('the %d , accuracy is %s, ce_loss is %s' % (epoch, accuracy, loss))
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
            # if self.should_save_logs:
            #     self.log_loss_accuracy(loss, accuracy, epoch, prefix='train')
            # if train_params.get('validation_set', False):
            #     print("Validation...")
            #     loss, acc = self.test(
            #         self.data_provider.validation, batch_size)
            #     if self.should_save_logs:
            #         self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            # time_per_epoch = time.time() - start_time
            # seconds_left = int((n_epochs - epoch) * time_per_epoch)
            # print("Time per epoch: %s, Est. complete in: %s" % (
            #     str(timedelta(seconds=time_per_epoch)),
            #     str(timedelta(seconds=seconds_left))))
            if epoch % 10000 == 0:
                self.saver.save(self.sess, './model_dense_0114/transfer_learn_%d' % epoch)
            # if self.should_save_model:
            #     self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    # def train_one_epoch(self, data, batch_size, learning_rate):
    #     num_examples = np.shape(data.train_imgs)[0]
    #     total_loss = []
    #     total_accuracy = []
    #     for i in range(num_examples // batch_size):
    #         images, labels = data.train_imgs[i * batch_size:(i + 1) * batch_size], \
    #                          data.train_labels[i * batch_size:(i + 1) * batch_size]
    #         feed_dict = {
    #             self.images: images,
    #             self.labels: labels,
    #             self.learning_rate: learning_rate,
    #             self.is_training: True,
    #         }
    #         fetches = [self.train_step, self.cross_entropy, self.accuracy]
    #         result = self.sess.run(fetches, feed_dict=feed_dict)
    #         _, loss, accuracy = result
    #         if i % 1 == 0:
    #             print('the %d , accuracy is %s, ce_loss is %s' % (i, accuracy, loss))
    #         total_loss.append(loss)
    #         total_accuracy.append(accuracy)
    #         if self.should_save_logs:
    #             self.batches_step += 1
    #             self.log_loss_accuracy(
    #                 loss, accuracy, self.batches_step, prefix='per_batch',
    #                 should_print=False)
    #     mean_loss = np.mean(total_loss)
    #     mean_accuracy = np.mean(total_accuracy)
    #     return mean_loss, mean_accuracy

    def test(self):
        self.saver.restore(self.sess, './model_vgg19_1127/transfer_learn_50000')
        qual_xt, unqual_xt, qual_yt, unqual_yt, test_files = vgg_train.load_test_data()
        # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
        xst = np.concatenate(qual_xt + unqual_xt, axis=0)
        yst = np.concatenate((qual_yt, unqual_yt), axis=0)
        state = np.random.get_state()
        np.random.shuffle(xst)
        np.random.set_state(state)
        np.random.shuffle(yst)
        np.random.set_state(state)
        np.random.shuffle(test_files)
        print('Net build')
        err_nums = 0
        for i, img_test in enumerate(xst):
            img0 = cv2.imread('../data/test/qual/' + test_files[i])
            if img0 is None:
                img0 = cv2.imread('../data/test/unqual/' + test_files[i])
                print('---------this pic is unqual')
            else:
                print('---------this pic is qual')
            if img0 is None:
                print('PIC NOT  FOUND')
            pre_out = self.sess.run((tf.nn.softmax(self.logits)), feed_dict={self.images: xst[i][np.newaxis, :],
                                                                             self.labels: yst[i][np.newaxis, :]})

            if pre_out[0][1] > 0.001:  # 此时判断为不合格
                if np.argmax(yst[i]) == 0:  # 真实值为合格
                    err_nums += 1
                    cv2.putText(img0, "WUJIAN Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1.0, (0, 0, 255), 2)
                    cv2.imwrite('../data/vggrst/i am wrong/qual/' + test_files[i], img0)
                else:
                    cv2.putText(img0, "GOOD Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1.0, (0, 0, 255), 2)
                    cv2.imwrite('../data/vggrst/i am right/' + test_files[i], img0)
            else:  # 判断合格！！
                if np.argmax(yst[i]) == 1:
                    err_nums += 1
                    cv2.putText(img0, "LOUJIAN Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1.0, (0, 0, 255), 2)
                    cv2.imwrite('../data/vggrst/i am wrong/unqual/' + test_files[i], img0)
                else:
                    cv2.putText(img0, "GOOD Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1.0, (0, 0, 255), 2)
                    cv2.imwrite('../data/vggrst/i am right/' + test_files[i], img0)


            print("the %d test image, predict is %s,ys is %s, error count is %d" % (i, pre_out, yst[i], err_nums))


        # num_examples = np.shape(data.test_imgs)[0]
        # total_loss = []
        # total_accuracy = []
        # for i in range(num_examples // batch_size):
        #     images, labels = data.test_imgs[i * batch_size:(i + 1) * batch_size], \
        #                      data.test_labels[i * batch_size:(i + 1) * batch_size]
        #     feed_dict = {
        #         self.images: images,
        #         self.labels: labels,
        #         self.is_training: False,
        #     }
        #     fetches = [self.cross_entropy, self.accuracy]
        #     loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
        #     total_loss.append(loss)
        #     total_accuracy.append(accuracy)
        #     print('the %d test, total loss is %s, total accuracy is %s' % (i, loss, accuracy))
        # mean_loss = np.mean(total_loss)
        # mean_accuracy = np.mean(total_accuracy)
        # return mean_loss, mean_accuracy
