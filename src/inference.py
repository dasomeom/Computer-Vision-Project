import argparse
import torch
from torchvision import datasets, transforms
from model import *
from train import *
import matplotlib.pyplot as plt
from imageio import imread, imsave

def infernece(use_cuda, model_path):
    device = torch.device("cuda" if use_cuda else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).items()
            plt.imshow(data.reshape(28, 28), cmap='Greys')
            plt.show()

def main():
    argparser = argparse.ArgumentParser(description='add args here')
    argparser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    argparser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    argparser.add_argument('--model_path', type=str, default="mnist_cnn.pt",
                        help='path of saved model')
    args = argparser.parse_args()

    if args.cuda:
        print("Using CUDA")
    use_cuda = not args.cuda and torch.cuda.is_available()
    model_path = args.model_path

    infernece(use_cuda, model_path)

if __name__ == '__main__':
    main()