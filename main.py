from DenseNet import DenseNet 
from utils import *
import argparse

def parse_args():
    description = "Tensorflow implementation of DensNet for cifar-like dataset"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--phase', type=str, default='train', help='train or test?')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100]')
    parser.add_argument('--epoch', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate, will be modified during training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save the checkpoints')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Directory to save the logs during training')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_convs', type=list, default=[6,6,6],
                        help='List of number of convolutions for each dense block')
    parser.add_argument('--compression', type=float, default=1.0,
                        help='Compression rate at the transition layer')
    parser.add_argument('--bottleneck', type=bool, default=True,
                        help='Using the bottleneck convolution or not')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Growth rate of the network')
    parser.add_argument('--batch_norm',type=bool, default=True,
                        help='Using the Batch Normalization or not')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='The parameter value of l2 regularization')
    return check_args(parser.parse_args())

def check_args(args):
    check_dir(args.checkpoint_dir)
    check_dir(args.logdir)
    try:
        assert args.epoch >=1
    except:
        print('the number of epochs must be larger than zero')
    try: 
        assert args.batch_size >=1
    except:
        print('the size of mini batch mus be larger than zero')
    return args

def main():
    args = parse_args()
    with tf.Session() as sess:
        network=DenseNet(sess, args)

        network.build_graph()
        
        show_all_variables()

        if args.phase == 'train':
            network.fit()
            print('Training finished!')

if __name__ == '__main__':
    main()