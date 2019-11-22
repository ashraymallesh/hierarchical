import time

import torch
from torch import nn, optim
from torchvision.transforms import transforms
from src.data import ModifiedMNISTDataset, train_test_split
from src.model import (SimpleCNN, pretrained_resnet18, pretrained_resnet50,
                       pretrained_resnet101, pretrained_wideresnet,
                       pretrained_mobilenet, pretrained_shufflenet, pretrained_shufflenet2)
from src.metrics import APRF

from fastai.vision import get_transforms, RandTransform, jitter, perspective_warp, skew, squish, symmetric_warp, tilt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from pathlib import Path
import pickle


def eval_model(model, criterion, dataloader, metric_helper, device):
    model.eval()
    for inputs, labels in tqdm(dataloader, desc=f"Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            # loss = criterion(outputs, labels)
            # running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            metric_helper.update_eval(outputs, labels, criterion)


def train_model(model, criterion, optimizer, scheduler,
                train_dataloader, valid_dataloader,
                metric_helper, tb_writer,
                device, num_epochs=25,
                batch_accum=1):
    since = time.time()
    best_acc = 0.0

    # begin training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        baccum_i = batch_accum # gradient accumulation
        model.train()
        optimizer.zero_grad()
        for inputs, labels in tqdm(train_dataloader, desc=f"Training ({epoch}/{num_epochs})"):
            baccum_i -= 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / batch_accum
            loss.backward()
            # desired number of batches accumulate
            if baccum_i == 0:
                baccum_i = batch_accum
                optimizer.step()
                optimizer.zero_grad()

            metric_helper.update_train(outputs, labels, criterion)
        # not enough batches to accumulate desired number
        if baccum_i != batch_accum:
            baccum_i = batch_accum
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        eval_model(model, criterion, valid_dataloader, metric_helper, device)

        summary = metric_helper.summary()

        train_loss = summary["train_loss"]
        train_acc = summary["train_a"]
        eval_loss = summary["eval_loss"]
        eval_acc = summary["eval_a"]
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
        print('Valid Loss: {:.4f} Acc: {:.4f}'.format(eval_loss, eval_acc))

        if eval_acc > best_acc:
            best_acc = eval_acc
            print(f"improved validation accuracy, saving model to {MODEL_SAVE_DIR}")
            torch.save(model.state_dict(), MODEL_SAVE_DIR / f"model.th")
        (MODEL_SAVE_DIR / f"metrics.pkl").write_bytes(pickle.dumps(metric_helper.epoch_summaries))
        for k,v in summary.items():
            tb_writer.add_scalar(tag=k, scalar_value=v, global_step=epoch)
        print()

    time_elapsed = time.time() - since
    torch.save(model.state_dict(), MODEL_SAVE_DIR / f"model_final.th")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return model


if __name__ == '__main__':
    DEBUG = False
    MODEL_TYPE = "WIDERESNET" # RESNET18, RESNET50, RESNET101, WIDERESNET, MOBILENET, SHUFFLENET, SHUFFLENET2, SIMPLE_CNN
    PRETRAINED = True
    TORCHVISION_TRANSFORM = False
    FASTAI_TRANSFORM = True
    TRAIN_TEST_SPLIT = 0.99
    BATCH_SIZE = 32
    BATCH_ACCUM = 4
    NUM_EPOCHS = 30
    IMAGE_RESIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EXPERIMENT_NAME = f"{MODEL_TYPE}_PT={PRETRAINED}_BS={BATCH_SIZE}_FAT={FASTAI_TRANSFORM}_IRS={IMAGE_RESIZE}"
    MODEL_SAVE_DIR = Path(f"saved_models/{EXPERIMENT_NAME}")
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    tb_writer = SummaryWriter(log_dir=f"saved_models/tensorboard/{EXPERIMENT_NAME}")

    if TORCHVISION_TRANSFORM:
        INPUT_DIM = 256
        tv_t = transforms.Compose([
                transforms.Resize(INPUT_DIM),
                transforms.ToTensor()])
    else:
        tv_t = None

    if FASTAI_TRANSFORM:
        fa_t_train, fa_t_eval = get_transforms(
            do_flip=False,
            max_rotate=25,
            xtra_tfms= [
                RandTransform(jitter, kwargs={'magnitude':(-0.02, 0.02)}, p=0.2, do_run=True, is_random=True),
                RandTransform(perspective_warp, kwargs={'magnitude': (-0.5, 0.5)}, p=0.2, do_run=True, is_random=True),
                RandTransform(squish, kwargs={}, p=0.2, do_run=True, is_random=True),
                RandTransform(skew, kwargs={'magnitude': (-0.5, 0.5), 'direction': (0,7)}, p=0.2, do_run=True, is_random=True),
                RandTransform(tilt, kwargs={'magnitude': (-0.5, 0.5),  'direction': (0,3)}, p=0.2, do_run=True, is_random=True),
            ]
        )
    else:
        fa_t_train, fa_t_eval = None, None

    print(f"insantiating model {MODEL_TYPE} with pretrained={PRETRAINED}...")
    if MODEL_TYPE == "RESNET18":
        MODEL = pretrained_resnet18(10, PRETRAINED)
        TO_RGB = True
    elif MODEL_TYPE == "RESNET50":
        MODEL = pretrained_resnet50(10, PRETRAINED)
        TO_RGB = True
    elif MODEL_TYPE == "RESNET101":
        MODEL = pretrained_resnet101(10, PRETRAINED)
        TO_RGB = True
    elif MODEL_TYPE == "MOBILENET":
        MODEL = pretrained_mobilenet(10, PRETRAINED)
        TO_RGB=True
    elif MODEL_TYPE == "WIDERESNET":
        MODEL = pretrained_wideresnet(10, PRETRAINED)
        TO_RGB=True
    elif MODEL_TYPE == "SHUFFLENET":
        MODEL = pretrained_shufflenet(10, PRETRAINED)
        TO_RGB=True
    elif MODEL_TYPE == "SHUFFLENET2":
        MODEL = pretrained_shufflenet2(10, PRETRAINED)
        TO_RGB=True
    elif MODEL_TYPE == "SIMPLE_CNN":
        MODEL = SimpleCNN(10)
        TO_RGB = False

    MODEL.to(DEVICE)

    CRITERION = nn.CrossEntropyLoss()
    # OPTIMIZER = optim.SGD(MODEL.parameters(), lr=0.001, momentum=0.9)
    OPTIMIZER = optim.Adam(MODEL.parameters())
    LR_SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=7, gamma=0.1)

    print("loading dataset...")
    if DEBUG:
        train_dataset = ModifiedMNISTDataset.from_files("data/debug_train_x",
                                                        "data/debug_train_y.csv",
                                                        to_rgb=TO_RGB,
                                                        torchvision_transform=tv_t,
                                                        fastai_transform=fa_t_train,
                                                        resize=IMAGE_RESIZE)
        valid_dataset = ModifiedMNISTDataset.from_files("data/debug_valid_x",
                                                        "data/debug_valid_y.csv",
                                                        to_rgb=TO_RGB,
                                                        torchvision_transform=tv_t,
                                                        fastai_transform=fa_t_eval,
                                                        resize=IMAGE_RESIZE)
    else:
        x = pd.read_pickle("data/train_max_x")
        y = pd.read_csv("data/train_max_y.csv")

        train_x, train_y, valid_x, valid_y = train_test_split(x, y, split=TRAIN_TEST_SPLIT)

        train_dataset = ModifiedMNISTDataset(train_x, train_y,
                                             to_rgb=TO_RGB,
                                             torchvision_transform=tv_t,
                                             fastai_transform=fa_t_train,
                                             resize=IMAGE_RESIZE)
        valid_dataset = ModifiedMNISTDataset(valid_x, valid_y,
                                             to_rgb=TO_RGB,
                                             torchvision_transform=tv_t,
                                             fastai_transform=fa_t_eval,
                                             resize=IMAGE_RESIZE)

    print("creating data loader...")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("beginning training...")
    labels = [str(i) for i in range(10)]
    metric_helper = APRF(labels)
    model = train_model(MODEL, CRITERION, OPTIMIZER, LR_SCHEDULER,
                        train_dataloader, valid_dataloader,
                        metric_helper, tb_writer,
                        DEVICE, num_epochs=NUM_EPOCHS,
                        batch_accum=BATCH_ACCUM)


    print("running test set predictions")
    MODEL.load_state_dict(torch.load(MODEL_SAVE_DIR/"model_final.th"))
    test_dataset = ModifiedMNISTDataset.from_files("data/test_max_x",
                                                   to_rgb=TO_RGB,
                                                    torchvision_transform=tv_t,
                                                    fastai_transform=fa_t_eval,
                                                   resize=IMAGE_RESIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    MODEL.eval()
    testset_predictions = []
    for inputs in tqdm(test_dataloader, desc=f"Evaluating"):
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            outputs = MODEL(inputs)
            _, preds = torch.max(outputs, 1)
            testset_predictions += preds.tolist()
    df = pd.DataFrame([{"Id": i, "Label": l} for i, l in enumerate(testset_predictions)])
    df.to_csv((MODEL_SAVE_DIR / "submission.csv"), index=False)

