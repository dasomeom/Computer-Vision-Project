import cv2

import numpy
import argparse
import torch
from torchvision import datasets, transforms
from model import *
from train import *
from webcam import *
from PIL import Image
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
            # cv2.imshow('sampling', img)
            img_out = img

            # need to convert img_out to a 28, 28, 1 binary image
            img_out = frameToBinary(img_out)

            img_out = transform(img_out)
            output = model(img_out)
            _, argmax = output.max(-1)

            print(argmax)
            counter = 0
        counter = counter + 1

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

def frameToBinary(img):
    # convert to binary, segment image
    # or get multiple image segments with possible numbers
    # need blob detection or something to determine if number exists
    # then apply directly

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (28,28))
    cv2.imshow('sampling', img)
    img = img.astype('float32')
    img = img / 255;
    img = img.reshape( 28, 28, 1)
    return img

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