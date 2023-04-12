import argparse
from dataset import DataSet


def main():
    parser = argparse.ArgumentParser(description='Normalizing dataset and prepare for training')

    parser.add_argument('-d', '--dataset', dest='dataset', help='Add dataset path, where will be saved trainable dataset ', required=True)
    parser.add_argument('-r', '--read-from', dest='data_path', help='Add data path', required=True)
    parser.add_argument('-p', '--project', dest='name', help='Add project name')

    args = parser.parse_args()

    dataset = DataSet(args)

    dataset.create()


if __name__ == '__main__':
    main()
