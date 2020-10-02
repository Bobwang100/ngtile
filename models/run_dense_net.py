import argparse
from models.dense_net import DenseNet
from models import vgg_train
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tensorflow.python.client import device_lib

print([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])


class DataProvider:
    """
    data_shape: img shape like [224,224,3]
    n_classes:  how many kinds of images
    data.next_batch : get the batch data for training or testing
    train, validation, test
    """

    def __init__(self):
        self.n_classes = 2
        self.data_shape = [224, 224, 1]
        # if args.train:
        #     qual_x, unqual_x, qual_y, unqual_y = vgg_train.load_data()
        # # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
        #     xs = np.concatenate(qual_x + unqual_x, axis=0)
        #     ys = np.concatenate((qual_y, unqual_y), axis=0)
        #     state = np.random.get_state()
        #     np.random.shuffle(xs)
        #     np.random.get_state(state)
        #     np.random.shuffle(ys)                            # shuffle the train and test images
        #     self.train_imgs, self.train_labels = xs[:60], ys[:60]
        # if args.test:
        #     qual_xt, unqual_xt, qual_yt, unqual_yt = vgg_train.load_data()
        #     # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
        #     xst = np.concatenate(qual_xt + unqual_xt, axis=0)
        #     yst = np.concatenate((qual_yt, unqual_yt), axis=0)
        #     state = np.random.get_state()
        #     np.random.shuffle(xst)
        #     np.random.get_state(state)
        #     np.random.shuffle(yst)                            # shuffle the train and test images
        #     self.test_imgs, self.test_labels = xst, yst


# class TestDataProvider:
#     """
#     data_shape: img shape like [224,224,3]
#     n_classes:  how many kinds of images
#     data.next_batch : get the batch data for training or testing
#     train, validation, test
#     """
#     def __init__(self):

train_params_set = {
    # 'batch_size': 64,
    'batch_size': 10,
    'n_epochs': 5000000,
    # 'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20000,  # epochs * 0.5
    'reduce_lr_epoch_2': 30000,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}
#
# train_params_svhn = {
#     'batch_size': 64,
#     'n_epochs': 40,
#     'initial_learning_rate': 0.1,
#     'reduce_lr_epoch_1': 20,
#     'reduce_lr_epoch_2': 30,
#     'validation_set': True,
#     'validation_split': None,  # you may set it 6000 as in the paper
#     'shuffle': True,  # shuffle dataset every epoch or not
#     'normalization': 'divide_255',
# }

#
# def get_train_params_by_name(name):
#     if name in ['C10', 'C10+', 'C100', 'C100+']:
#         return train_params_cifar
#     if name == 'SVHN':
#         return train_params_svhn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet',
        help='What type of model to use')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 40],
        default=12,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int, choices=[40, 100, 190, 250],
        default=40,
        help='Depth of whole network, restricted to paper choices')
    # parser.add_argument(
    #     '--dataset', '-ds', type=str,
    #     choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],
    #     default='C10',
    #     help='What dataset should be used')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=3, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, default=0.8, metavar='',
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')

    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')

    parser.add_argument(
        '--num_inter_threads', '-inter', type=int, default=1, metavar='',
        help='number of inter threads for inference / test')
    parser.add_argument(
        '--num_intra_threads', '-intra', type=int, default=128, metavar='',
        help='number of intra threads for inference / test')

    parser.set_defaults(renew_logs=True)

    args = parser.parse_args(['--train'])
    # args = parser.parse_args(['--test'])
    # args = parser.parse_args(['--test', '--train'])

    # if not args.keep_prob:
    #     if args.dataset in ['C10', 'C100', 'SVHN']:
    #         args.keep_prob = 0.8
    #     else:
    #         args.keep_prob = 1.0
    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 1.0
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = train_params_set
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))
    print("Prepare training data...")
    data_provider = DataProvider()
    print("Initialize the model..")
    model = DenseNet(data_provider=data_provider, **model_params)

    if args.train:
        # print("Data provider train images: ", np.shape(data_provider.train_imgs))
        print("Data provider train images: ")
        model.train_all_epochs(train_params)
    if args.test:
        # model.load_model()
        # print("Data provider test images: ", np.shape(data_provider.test_imgs))
        print("Testing...")
        model.test()
        # model.test(data_provider, batch_size=10)
        # loss, accuracy = model.test(data_provider, batch_size=200)
        # print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
