import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on WRF and ERA fields')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--beta', type=float, default=0.5, help='beta parameter for loss function')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--model', '-m', type=str, default='trajGRU',
                        help='Choose model architecture: trajGRU, convLSTM, conv2d or BERTunet')
    parser.add_argument('--scheduler', type=str, default='MultiStepLR',
                        help='Choose scheduler type: MultiStepLR, linearWarmUp')
    parser.add_argument('--use-spatiotemporal-encoding', '-s', default=1, type=int,
                        help='If use spatiotemporal encoding')
    parser.add_argument('--use-time-encoding', '-t', default=1, type=int,
                        help='If use spatiotemporal encoding')
    parser.add_argument('--input-sequence-length', default=4, type=int, metavar='ISL',
                        help='input data sequence length')
    parser.add_argument('--output-sequence-length', default=4, type=int, metavar='zoSL',
                        help='output data sequence length')
    parser.add_argument('--run-mode', '-r', dest='run_mode', type=str, default='train+test',
                        help='run mode: train, test or train+test')
    parser.add_argument('--run-id', dest='run_id', type=int, default=4,
                        help='if run mode is test select run id')
    parser.add_argument('--best-epoch-id', dest='best_epoch_id', type=int, default=15,
                        help='if run mode is test select best epoch id')
    parser.add_argument('--draw_plots', type=int, default=1,
                        help='If draw resulting plots')
    parser.add_argument('--running-env', type=str, default='home',
                        help='Specify where you run the script: docker, io, home')
    return parser.parse_args()
