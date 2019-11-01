from sklearn.neighbors import KNeighborsClassifier
import scipy.io as scio
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import svhn.io as svhn

TRAINING = 0
TEST = 1
EXTRA = 2
N = 10000
#transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#
#trainset = torchvision.datasets.SVHN(root='../data', split='train', transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, shuffle=True)
#
#
#testset = torchvision.datasets.SVHN(root='../data', split='test', transform=transform)
#testloader = torch.utils.data.DataLoader(testset, shuffle=True)


if __name__ == "__main__":
    neighbor = KNeighborsClassifier(n_neighbors=10)
    labels = np.array([])
    input_data = np.array([[]])

    svhn_loader = svhn.SVHN()
    svhn_loader.load_cropped()
    
    X, y = svhn_loader.get_cropped_dataset(TRAINING)
    print("loaded")
    print("X: ", type(X))
    print("X ", X.shape)
    i  = 0
    for i in range(len(X[0,0,0,:])):
        #print(i)
        #print(X[:,:,:,i].shape)
        inp = np.array([X[:,:,:,i].flatten()])
        #print(inp.shape)
        if i == 0:
            input_data = np.array([X[:,:,:,i].flatten()])
            labels = np.array(y[i])
        else:
            input_data = np.concatenate((input_data, inp), axis=0)
            labels = np.concatenate((labels, np.array(y[i])))
        #print(input_data.shape)

    print("fit")
    print("input size: ", input_data.shape)
    print("labels train: ", labels)
    neighbor.fit(input_data, labels)
    labels_t = np.array([])
    input_data_t = []
    X, y = svhn_loader.get_cropped_dataset(TEST)
    for i in range(len(X[0,0,0,:])):
        inp = np.array([X[:,:,:,i].flatten()])
        if i == 0:
            input_data_t = np.array([X[:,:,:,i].flatten()])
            labels_t = np.array(y[i])
        else:
            input_data_t = np.concatenate((input_data_t, inp), axis=0)
            labels_t = np.concatenate((labels_t, np.array(y[i])))

    print(input_data_t.shape)
    print("labels test: ",len(labels_t))
    score = neighbor.score(input_data_t, labels_t)
    print("score: ", score)



