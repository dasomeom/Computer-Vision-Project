import cv2
import numpy as np

import argparse
import torch
from torchvision import datasets, transforms
from model import *
from train import *
from webcam import *
import matplotlib.pyplot as plt


# gets webcam data and displays it
samplingRate = 10


def webcam(use_cuda, model_path, mirror=False, ):
    cam = cv2.VideoCapture(0)
    counter = 0;

    device = torch.device("cuda" if use_cuda else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    while True:
        ret_val, img = cam.read()
        #img is an rgb array
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        # add sampling rate
        if counter == samplingRate:
            cv2.imshow('sampling', img)
            img_out = img
            counter = 0
        counter = counter + 1

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


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

    #infernece(use_cuda, model_path)
    webcam(use_cuda, model_path)

if __name__ == '__main__':
    main()