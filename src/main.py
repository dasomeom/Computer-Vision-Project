import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import *
from train import *
from test import *
from time import time
import math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description='add args here')
    argparser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    argparser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    argparser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    argparser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    argparser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    argparser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    argparser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    argparser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    argparser.add_argument('--save_model', type=bool, default=True, 
                        help='save model')
    args = argparser.parse_args()

    if args.cuda:
        print("Using CUDA")
    use_cuda = not args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    start = time()

    try:
        for epoch in range(1, args.epochs + 1):
            print("Training for %d epochs..." % args.epochs)
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

        if (args.save_model):
            torch.save(model.state_dict(), "mnist_cnn.pt")

    except KeyboardInterrupt:
        print("Saving before quit...")
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()