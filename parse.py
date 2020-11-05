import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ml-1m", help="choose dataset from ['ml-1m', 'pinterest-20']",
                    choices=['ml-1m', 'pinterest-20'])
parser.add_argument("--model", type=str, default="NeuMF-end", help="choose model from ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']",
                    choices=['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre'])
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--save", default=True, help="save model or not")

args = parser.parse_args()
