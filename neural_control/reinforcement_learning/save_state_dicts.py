from argparse import ArgumentParser
import os
import torch


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', type=str, help='path to folder leading to models')
    args = parser.parse_args()

    files = [f for f in os.listdir(args.path) if '.pth' in f]

    for file in files:
        model_path = os.path.join(args.path, file)
        assert model_path[-4:] == '.pth'

        print(model_path[:-1])
        model = torch.load(model_path)
        torch.save(model.state_dict(), model_path[:-1])
