'''
Contains following functions:

get_mis_classified_byloader(model, device, data_loader) - accepts input as data loader
get_miss_classified_byimages(model, device, images, labels) - accepts images as a tensor of shape (B, C, H, W) 
getFractionsMissed - to retrieve fractions missed by each class in the dataset
plot_misclassified(image_data, targeted_labels, predicted_labels, classes, no_images) - to plot mis classified images
plot_LossAndAcc - plot train/test loss and accuracies
get_mean_and_std - returns mean and std of the dataset
denormalize(image) - to denormalize an image
imshow(img) - to plot an image
'''

### Functions to plot misclassified images

'''
function to return data points for plotting misses
It accepts the following inputs:
1. trained model - trained on GPU, 2. device - cuda device, 3. images as a tensor (channel*shape*shape),
4) labels as 1D torch tensor

Outputs provided for all incorrect predictions, where length of each list = number of batches:
1. data_images - List of torch tensors of images, if no mispredictions in a given batch then empty tensor
2. pred_labels - List of Predicted labels, if no mispredictions in a given batch then empty tensor
3. target_labels - Ground truth or target labels, if no mispredictions in a given batch then empty tensor
'''
import torch
import collections
import matplotlib.pyplot as plt

'''
function to return data points for plotting misses
It accepts trained model, device and test_loader as inputs
Outputs provided for all incorrect predictions, where length of each list = number of batches:
1. data_images - List of torch tensors of images, if no mispredictions in a given batch then empty tensor 
2. pred_labels - List of Predicted labels, if no mispredictions in a given batch then empty tensor 
3. target_labels - Ground truth or target labels, if no mispredictions in a given batch then empty tensor 
'''


def get_mis_classified_byloader(model, device, data_loader):
    model.eval()
    missed_images = []  # will contain list of batches, for a given batch will return list of indices not predicted correctly
    # empty list will indicate no mis predictions
    pred_labels = []  # contains list of predicted labels by each batch
    data_images = []  # contains list of images by each batch for plotting
    target_labels = []  # contains list of target labels by batch
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # missed_images.append(torch.where(torch.not_equal(pred.squeeze(),target).cpu()))
            misses = torch.where(torch.not_equal(pred.squeeze(), target))
            data_images.append(data[misses].cpu())
            target_labels.append(target[misses].cpu())
            pred_labels.append(pred[misses].cpu())

    pred_labels = [x.item() for item in pred_labels for x in item]
    target_labels = [x.item() for item in target_labels for x in item]
    data_images = [x for item in data_images for x in item]

    return data_images, pred_labels, target_labels



def get_miss_classified_byimages(model, device, images, labels):
    #images to labels to GPU
    if not (images.is_cuda):
        images = images.to(device)
    if not (labels.is_cuda):
        labels = labels.to(device)

    model.eval()

    with torch.no_grad():
        output = model(images)
        #predict labels
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #identify positions that are misclassified
        misses = torch.where(torch.not_equal(pred.squeeze(), labels))[0].cpu()

        # get missed information convert tensors back to cpu() for plotting images
        if images.is_cuda:
            data_images = images[misses].cpu()
        if labels.is_cuda:
            target_labels = labels[misses].cpu()
        if pred.is_cuda:
            pred_labels = pred[misses].cpu()

    return data_images, target_labels, pred_labels

'''
Function getFractionsMissed --> 
Input: model, device, test_loader
Output: fractions missed by each class
'''
def getFractionsMissed(model, device, test_loader):
    model.eval()
    missed_targets = []  # contains list of target labels by batch
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index

            misses = torch.where(torch.not_equal(pred.squeeze(), target))
            missed_targets.append(target[misses].cpu())
            targets.append(target.cpu())

    targets = [x.item() for item in targets for x in item]
    missed_targets = [x.item() for item in missed_targets for x in item]

    # using Counter to find frequency of elements
    target_freq = collections.Counter(targets)
    miss_freq = collections.Counter(missed_targets)

    # calculate fraction misses for each class
    fractions_dict = {key: miss_freq[key] / target_freq.get(key, 0)
                      for key in target_freq.keys()}
    # sort fractions_dict by keys
    keys = list(fractions_dict.keys())
    keys.sort()
    fractions_dict = {i: fractions_dict[i] for i in keys}

    return fractions_dict


def plot_misclassified(image_data, targeted_labels, predicted_labels, classes, no_images):
    no_images = min(no_images, len(predicted_labels))

    figure = plt.figure(figsize=(12, 5))

    for index in range(1, no_images + 1):
        image = denormalize(image_data[index - 1]).numpy().transpose(1, 2, 0)
        plt.subplot(2, 5, index)

        plt.imshow(image)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        title = "Target:" + str(classes[targeted_labels[index - 1]]) + "\nPredicted:" + str(
            classes[predicted_labels[index - 1]])
        plt.title(title)


def plot_LossAndAcc(train_acc,train_losses,test_acc,test_losses):

    # function for plotting test and training loss/ accuracy
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    axs[0].plot(train_losses, label='Training Losses')
    axs[0].plot(test_losses, label='Test Losses')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title("Loss")

    axs[1].plot(train_acc, label='Training Accuracy')
    axs[1].plot(test_acc, label='Test Accuracy')
    axs[1].legend(loc='lower right')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title("Accuracy")

    plt.show()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T

# functions to denormalize image

def denormalize(image):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    inv_transform= T.Compose([
          T.Normalize(
              mean = (-1 * np.array(mean) / np.array(std)).tolist(),
              std = (1 / np.array(std)).tolist()
          )])

    return inv_transform(image)

#function to show image
def imshow(img):
    img = denormalize(img)     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))