import time
import copy

import torch
from torch import nn, optim
from torchvision.transforms import transforms
from src.data import ModifiedMNISTDataset, train_test_split
from src.model import SimpleCNN, pretrained_resnet18
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd


def eval_model(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(dataloader, desc=f"Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(dataloader) / dataloader.batch_size
    eval_acc = running_corrects.double() / len(dataloader) / dataloader.batch_size

    return {"loss": eval_loss, "acc": eval_acc}


def train_model(model, criterion, optimizer, scheduler,
                train_dataloader, valid_dataloader,
                device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_dataloader, desc=f"Training ({epoch}/{num_epochs})"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader) / train_dataloader.batch_size
        epoch_acc = running_corrects.double() / len(train_dataloader) / train_dataloader.batch_size

        valid_results = eval_model(model, criterion, valid_dataloader, device)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print('Valid Loss: {:.4f} Acc: {:.4f}'.format(valid_results["loss"],
                                                      valid_results["acc"]))

        if valid_results["acc"] > best_acc:
            best_acc = valid_results["acc"]
            best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    DEVICE = torch.device("cpu")
    # MODEL = SimpleCNN(256, 10)
    # TO_RGB = False
    MODEL = pretrained_resnet18(10)
    TO_RGB = True
    CRITERION = nn.CrossEntropyLoss()
    # OPTIMIZER = optim.SGD(MODEL.parameters(), lr=0.001, momentum=0.9)
    OPTIMIZER = optim.Adam(MODEL.parameters())
    LR_SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=7, gamma=0.1)
    BATCH_SIZE = 4


    t = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()])

    # using entire dataset (6hrs/epoch on my laptop)
    x = pd.read_pickle("../data/train_max_x")
    y = pd.read_csv("../data/train_max_y.csv")

    train_x, train_y, valid_x, valid_y = train_test_split(x, y)

    train_dataset = ModifiedMNISTDataset(train_x, train_y,
                                         to_rgb=TO_RGB, transform=t)
    valid_dataset = ModifiedMNISTDataset(valid_x, valid_y,
                                         to_rgb=TO_RGB, transform=t)

    # train_dataset = ModifiedMNISTDataset.from_files("../data/debug_train_x",
    #                                                 "../data/debug_train_y.csv",
    #                                                 to_rgb=False, transform=t)
    # valid_dataset = ModifiedMNISTDataset.from_files("../data/debug_valid_x",
    #                                                 "../data/debug_valid_y.csv",
    #                                                 to_rgb=False, transform=t)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_model(MODEL, CRITERION, OPTIMIZER, LR_SCHEDULER, train_dataloader, valid_dataloader, DEVICE)