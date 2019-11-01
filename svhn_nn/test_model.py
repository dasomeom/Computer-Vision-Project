import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from svhn_nn.model import Net


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.SVHN(root='../../data', split='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


PATH = './svhn_net_2.pth'
net = Net()
net.load_state_dict(torch.load(PATH))


correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
running_loss = 0.0
loss_epoch_array = np.zeros((1, 2))


with torch.no_grad():
    for epoch in range(10):
        for i, data in enumerate(testloader):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()




            loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                loss_epoch_array = np.concatenate((loss_epoch_array, np.array([[epoch + 1, running_loss / 2000]])))
                running_loss = 0.0
        print('Accuracy of the network on the {0} test images: {1}'.format(total,
            100 * correct / total))

np.save("loss_epoc_array_test", loss_epoch_array)
