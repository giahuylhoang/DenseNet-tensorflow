import tensorflow as tf




def new_weights(shape, name):
    return tf.get_variable(name, 
                            shape,
                            initializer = tf.contrib.layers.xavier_initializer())


def new_biases(lenth, name):
    return tf.get_variable(name,
                            [lenth],
                            initializer = tf.zeros_initializer())


def conv2d(input,
            num_filters,
            strides,
            kernel_size,
            scope):
    num_input_chanels = input.get_shape().as_list()[-1]
    with tf.name_scope(scope):
        shape = [kernel_size, 
                 kernel_size, 
                 num_input_chanels, 
                 num_filters]
        weights = new_weights(shape, 
                              'weights_')# + scope)
        biases = new_biases(num_filters, 
                            'biases_')# + scope)
        layer = tf.nn.conv2d(input = input,
                             filter = weights,
                             strides = [1,strides,strides,1],
                             padding='SAME') + biases
    return layer


def batch_normalization(input,
                        scope):
    shape = input.get_shape().as_list()[1:4]
    with tf.name_scope(scope):
        batch_mean, batch_var = tf.nn.moments(input, 
                                                axes=[0],
                                                keep_dims=False)
        gamma = tf.Variable(name = 'gamma_', # + scope,
                            initial_value = tf.ones(shape))
        beta = tf.Variable(name ='beta_', # + scope,
                            initial_value = tf.zeros(shape))
        batchnorm = tf.nn.batch_normalization(input,
                                                mean = batch_mean,
                                                variance = batch_var,
                                                offset = beta,
                                                scale = gamma,
                                                variance_epsilon = 1.001e-5)
        return batchnorm


def dense_layer(input,
                output_shape,
                scope):
    with tf.name_scope(scope):
        input_shape = input.get_shape().as_list()[-1]
        weights = new_weights([input_shape,
                                output_shape],
                                'weight_')# + scope)
        biases = new_biases(output_shape,
                            'biase_')# + scope)
        layer = tf.matmul(input, weights) + biases
    return layer


def reLU(input,
         scope):
        with tf.name_scope(scope):
            activation = tf.nn.relu(input)
        return activation


def flatten_layer(input,
                  scope):
    with tf.name_scope(scope):
        width = input.get_shape()[1:4].num_elements()
        output_shape = [-1, width]
    return tf.reshape(input, output_shape)


def global_average_pooling(input,
                            scope):
    with tf.name_scope(scope):
        input_shape = input.get_shape().as_list()
        strides = ksize = [1] + input_shape[1:3] + [1]
        pooling = tf.nn.avg_pool(input,
                                    ksize=ksize,
                                    strides=strides,
                                     padding='VALID')
    return pooling


def max_pooling(input, 
                scope,
                strides = 2,
                padding = 'VALID'):
    with tf.name_scope(scope):
        ksize = strides = [1, strides, strides, 1]
        pooling = tf.nn.max_pool(input,
                                    ksize=ksize,
                                    strides=strides,
                                    padding=padding)
    return pooling
def average_pooling(input, 
                    scope,
                    strides = 2,
                    padding = 'VALID'):
    with tf.name_scope(scope):
        ksize = strides = [1, strides, strides, 1]
        pooling = tf.nn.avg_pool(value=input,
                                    ksize=ksize,
                                    strides=strides,
                                    padding=padding)
def dense_block(name,
                    input,
                    num_conv,
                    growth_rate,
                    kernel_size,
                    batch_normalization,
                    bottleneck):
    concat = input
    layer = input
    for l in range(num_conv):
        if batch_normalization:
            layer = batch_normalization(concat,
                                        name + '_BN_conv_' + str(l))
        layer = reLU(concat,
                     'activation_' + str(num_conv) + str(l))
        if bottleneck: 
            layer = conv2d(input=layer,
                           num_filters=growth_rate*4,
                           strides=1,
                           kernel_size=1,
                           scope=name + '_conv_bottleneck_' + str(l))
            if batch_normalization:
                    layer = batch_normalization(layer,
                                                name+'_BN_bottleneck_' + str(l))
        layer = reLU(layer,
                     scope='activation_bottleneck_' + str(num_conv) + str(l))
        layer = conv2d(input=layer,
                       num_filters=growth_rate,
                       strides=1,
                       kernel_size=kernel_size,
                       scope=name + '_conv_' + str(l))
        concat = tf.concat([concat, layer],
                           axis = 3,
                           name = name + '_concat_' + str(l))
        return concat

                                                
def transition_layer(name,
                    input,
                    compression=1.0):
    num_channels = input.get_shape().as_list()[-1]
    num_filters = int(num_channels*compression)
    layer = conv2d(input=input,
                   num_filters=num_filters,
                   strides=1,
                   kernel_size=1,
                   scope=name)
    layer = average_pooling(input=layer,
                            strides=2,
                            scope=name + '_avg_pool_')
    return layer


def loss(logits, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, 
                                                                     labels=labels))
    return loss


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=0), 
                                  tf.argmax(labels, axis=0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
