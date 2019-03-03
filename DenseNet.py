import tensorflow as tf
import time
from ops import *
from utils import *

class DenseNet(object):
    def __init__(self, sess, args):
        self.model_name = 'DenseNet_cifar'
        self.sess = sess
        self.dataset_name = args.dataset
        if self.dataset_name == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10()
            self.input_shape = [32, 32, 3]
            self.num_classes = 10
        if self.dataset_name == 'cifar100':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar100()
            self.input_shape = [32, 32, 3]
            self.num_classes = 100
        self.growth_rate = args.growth_rate
        self.num_convs = args.num_convs
        self.depth = (len(args.num_convs) + 
                      (1+int(args.bottleneck))*
                      np.sum(args.num_convs) + 1)
        self.bottleneck = args.bottleneck
        self.compression = args.compression
        self.kernel_size = args.kernel_size
        self.batch_norm = args.batch_norm
        self.weight_decay = args.weight_decay
        self.logdir = args.logdir
        self.checkpoint_dir = args.checkpoint_dir
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.x_train) // self.batch_size
        self.init_lr = args.lr
    def model(self,
              input,
              reuse=False):
        with tf.variable_scope('DenseNet', reuse=reuse):
            layer = conv2d(input,
                           num_filters = self.growth_rate*2,
                           strides = 1,
                           kernel_size = self.kernel_size,
                           scope = 'init_conv_')
            #Dense Block
            for num_conv, lv in zip(self.num_convs,range(len(self.num_convs))):
                layer = dense_block(name='dense_lv_' + str(lv),
                                    input=layer,
                                    num_conv=num_conv,
                                    growth_rate=self.growth_rate,
                                    kernel_size=self.kernel_size,
                                    batch_norm=self.batch_norm,
                                    bottleneck=self.bottleneck)
        #Transition Layer
                if lv < len(self.num_convs) - 1:
                    layer = transition_layer(name='trans_lv_' + str(lv),
                                             input=layer,
                                             compression=self.compression)
            layer = global_average_pooling(layer,
                                       scope='global_avg_pool')
            layer = flatten_layer(layer,
                              scope='flattened')
            logits = dense_layer(layer,
                             self.num_classes,
                             scope='dense')
        return logits
    def build_graph(self):
        self.inputs = tf.placeholder(tf.float32, [] + self.input_shape , name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name='labels')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.logits = self.model(self.inputs)
        with tf.name_scope('loss'):
            self.loss = loss(logits=self.logits,
                            labels=self.labels)
        with tf.name_scope('accuracy'):
            self.accuracy = accuracy(logits=self.logits,
                                    labels=self.labels)
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.train_loss_summary = tf.summary.scalar('train_loss', self.loss)
        self.test_loss_summary = tf.summary.scalar('test_loss', self.loss)
        self.train_acc_summary = tf.summary.scalar('train_accuracy', self.accuracy)
        self.test_acc_summary = tf.summary.scalar('test_accuracy', self.accuracy)
        self.train_summary = tf.summary.merge([self.train_loss_summary, self.train_acc_summary])
        self.test_summary = tf.summary.merge([self.test_loss_summary, self.test_acc_summary])
    def fit(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(os.path.join(self.logdir, self.model_dir), self.sess.graph)
        loaded, checkpoint_counter = self.load(self.checkpoint_dir)
        if loaded == True: 
            epoch_lr = self.init_lr 
            start_epoch = int(checkpoint_counter/self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter 
            
            if start_epoch >= int(self.epoch * 0.75):
                epoch_lr = epoch * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and star_epoch < int(self.epoch * 0.75):
                epoch_lr = epoch * 0.1
            print('Successfully load saved model')
        else:
            epoch_lr = self.init_lr
            start_epoch = 0 
            start_batch_id = 0 
            counter = 1
            print('Fail to load the saved model')
        start_time = time.time()
        x_train = self.x_train
        y_train = self.y_train
        for epoch in range(start_epoch, self.epoch):
            if start_epoch == 0: 
                x_train, y_train = shuffle(x_train, y_train)
            for index in range(start_batch_id, self.iteration):
                batch_x = x_train[index*self.batch_size:(index+1)*self.batch_size]
                batch_y = y_train[index*self.batch_size:(index+1)*self.batch_size]

                _, summary_train, train_loss, train_accuracy = self.sess.run([self.optimizer,
                                                                            self.train_summary,
                                                                            self.loss,
                                                                            self.accuracy],
                                                                            feed_dict = 
                                                                            {self.inputs: batch_x,
                                                                             self.labels: batch_y,
                                                                             self.lr: epoch_lr})

                self.writer.add_summary(summary_train, counter)
                summary_test, test_loss, tess_accuracy = self.sess.run([self.test_summary,
                                                                        self.loss,
                                                                        self.accuracy],
                                                                        feed_dict = 
                                                                        {self.inputs: self.x_test,
                                                                         self.labels: self.y_test})
                self.writer.add_summary(summary_train, counter)
                print('Epoch: {0:} {1:}, learning_rate {2: .4f}'.format(epoch, index, epoch_lr))
                print('time: {0: .4f}, train_accuracy {1: .4f}, test_accuracy {2: 4f}'.format(time.time()- start_time,
                                                                                              train_accuracy, 
                                                                                              test_accuracy))
                counter += 1
            start_batch_id = 0
            self.save(self.checkpoint_dir, counter)
        self.save(self.checkpoint_dir, counter)
    def test(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        loaded, checkpoint_counter = self.load(self.checkpoint_dir)
        if loaded:
            print('Load succesully')
        else:
            print('Load failed')
        test_loss, tess_accuracy = self.sess.run([self.loss,
                                                  self.accuracy],
                                                  feed_dict = {self.inputs: self.x_test,
                                                               self.labels: self.x_test})
        print('test_loss: {0: .5f}  test_accuracy: {1: .4f}'.format(test_loss, test_accuracy))
    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.depth, self.dataset_name, self.batch_size, self.init_lr)

    def load(self, checkpoint_dir):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            counter = ckpt_name.split('-')[-1]
            print('Successfully load {}'.format(ckpt_name))

            return True, counter
        else:
            print('There are no existed checkpoint files or directory')
            return False, 0

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if os.path.exists(checkpoint_dir) == False:
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name+'.model', global_step=step))