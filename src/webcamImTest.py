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
samplingRate = 20


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
        img = img[80:400, 100:540]
        #cv2.imshow('cropped', img)
        #img = cv2.Canny(img, 80, 200)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', img)

        # add sampling rate
        if counter == samplingRate:
            # cv2.imshow('sampling', img)
            temp_img = img
            list = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
            sum = 48
            sha_0, sha_1 = img.shape
            for i in range(3): # 0 - 160, 80 - 240, 160 - 320
                for j in range(3): #
                    #print(i , j)
                    window = img[i * int(sha_0/3):i * int(sha_0/3) + int(sha_1/2), j * int(sha_1/3): j * int(sha_1/3) + int(sha_1/2)]
                    #print(window.shape, " window")
                    img_out = cv2.resize(window, (28,28))
                    img_out = img_out.astype('float32')
                    img_out = 1 - (img_out / 255)
                    img_out = img_out.reshape(28, 28, 1)

                    img_out = transform(img_out)
                    img_out = img_out[None]
                    plt.imshow(img_out.reshape(28, 28), cmap='Greys')
                    output = model(img_out)
                    _, argmax = output.max(-1)
                    am = int(str(argmax).split("[")[1][0])
                    list[am] += 1
            # need to convert img_out to a 28, 28, 1 binary image
            print(list)
            #print([k for k,v in list.items() if v >= (sum * .3)])
            counter = 0
        counter += 1

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

def frameToBinary(img):
    # convert to binary, segment image
    # or get multiple image segments with possible numbers
    # need blob detection or something to determine if number exists
    # then apply directly

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    img = cv2.resize(img, (28,28))
    cv2.imshow('sampling', img)
    img = img.astype('float32')

    img = img /255
    img = 1 - img
    img = img.reshape(28, 28, 1)
    cv2.imshow('sampling', img)
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