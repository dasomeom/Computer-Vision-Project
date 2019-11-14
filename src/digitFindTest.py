import cv2
import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans
from skimage import img_as_bool
from skimage.transform import resize
from scipy import signal
import matplotlib.pyplot as plt
import argparse
import torch
from torchvision import datasets, transforms
from model import *
from train import *
from webcam import *

def kMeansApproach(img):
    #Mx3
    size = np.shape(img)
    val = np.reshape(img, size[0]*size[1], size[2])
    clusters, label = scipy.cluster.vq.kmeans2(val, 2)
    np.reshape(label, size[0], size[1], size)
    outputIm = np.zeros(shape=(size[0],size[1],size[2]))
    for i in range(0,size[0]):
        for j in range(0, size[1]):
            lab = label[i*j + j]
            outputIm[i][j][0] = clusters[lab][0]
            outputIm[i][j][1] = clusters[lab][1]
            outputIm[i][j][2] = clusters[lab][2]
    plt.figure()
    plt.imshow(outputIm)
def kMeansActual(img):
    # load the image and grab its width and height

    image = img
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=2)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    lab = labels.reshape((h,w,1))
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    quant = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
    #plt.figure()
    #plt.imshow(lab, cmap='gray')
    plt.show()
    resized = img_as_bool(resize(labels.astype("bool"), (28, 28)))
    #when resized, depending on number, it looses info
    resized = resized.reshape(28, 28, 1)
    resized = resized.astype('float32')
    return resized

def edgeDetection(img, lineThickness):
    # Padded fourier transform, with the same shape as the image
    # We use :func:`scipy.signal.fftpack.fft2` to have a 2D FFT
    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((2, lineThickness), np.uint8)

    img_dilation = cv2.dilate(edges, kernel, iterations=5)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=5)
    plt.figure()
    plt.imshow(img_erosion, cmap='gray')
    plt.figure()
    plt.imshow(img_dilation, cmap='gray')
    return img_erosion

def frameToBinary(img):
    # convert to binary, segment image
    # or get multiple image segments with possible numbers
    # need blob detection or something to determine if number exists
    # then apply directly

    img = cv2.resize(img, (28,28))
    cv2.imshow('sampling', img)
    img = img.astype('float32')

    img = img /255
    img = 1 - img
    img = img.reshape(28, 28, 1)
    cv2.imshow('sampling', img)
    return img


def main():
    # import test image
    # find bounding box

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


    device = torch.device("cuda" if use_cuda else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    #path = '4-and-9.jpg' test images
    #path = '5.png'
    img = cv2.imread(path)

    #img_out = edgeDetection(img, 10)
    img_out = kMeansActual(img)

    #img_out = frameToBinary(img_out)
    cv2.imshow('sampling', img_out)
    img_out = transform(img_out)
    img_out = img_out[None];
    plt.imshow(img_out.reshape(28, 28), cmap='Greys')
    output = model(img_out)
    _, argmax = output.max(-1)
    print(argmax)


if __name__ == '__main__':
    main()