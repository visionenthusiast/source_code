from tqdm import tqdm
import torch

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  # Import your choice of scheduler here

def train_model(model, train_loader, test_loader, device, epochs=2):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    test_losses = []
    train_acc_all = []
    test_acc_all = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

        train_acc = 100. * correct / len(train_loader.dataset)

        #accumulate training losses & accuracy for plotting
        train_losses.append(running_loss/len(train_loader))
        train_acc_all.append(train_acc)

        print("Epoch: ", epoch, "Learning Rate: ", optimizer.param_groups[0]["lr"])
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss, correct, len(train_loader.dataset), train_acc))

        test_loss, test_acc = test(model, device, test_loader)

        # accumulate test losses & accuracy for plotting
        test_losses.append(test_loss)
        test_acc_all.append(test_acc)

        # increment learning rate
        scheduler.step()

    print('Finished Training')

    return model, train_losses, train_acc_all, test_losses, test_acc_all

def test(model, device, test_loader):
    '''
    test function taken inputs as pre defined model, device, test_loader (dataloader)
    It returns the following: test loss, test accuracy for that epoch
    '''
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100. * correct / len(test_loader.dataset)

    # return test loss and test accuracy
    return loss, acc
