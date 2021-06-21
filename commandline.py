
import sys
import getopt
from configs import data_config, train_config


def parse_args(argv):
    try:
        opts, args = getopt.getopt(
            argv, "t:e:d:w:", ["trainbs=", "epochs=", "datadir=", "workers="])
    except getopt.GetoptError:
        print(
            'main.py -t <train_batch_size> -e <epochs> -d <datadir> -w <workers>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'main.py -t <train_batch_size> -e <epochs> -d <datadir> -w <workers>')
            sys.exit()
        elif opt in ("-t", "--trainbs"):
            data_config.train_batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            train_config.epochs = int(arg)
        elif opt in ("-d", "--datadir"):
            data_config.data_dir = arg
        elif opt in ("-w", "--workers"):
            data_config.num_workers = int(arg)
