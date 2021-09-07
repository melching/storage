# Some simple script demonstrating a way to train an image classifier
# more detailed: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm #pretrained backbones

import numpy as np

import tqdm


# some settings needed later on
NUM_CLASSES = 2
DEVICE = torch.device("cuda:0")
BATCH_SIZE = 64

EPOCHS = 30
EVAL_EVERY_N_EPOCHS = 2


# define train/val loader
train_loader = None
val_loader = None


# you could just use timm.create to init the model, but as I might want to do something else later
# I create an extra class
class SimpleImageClassifier(nn.Module):
    def __init__(self, num_classes):
        self.super().__init__()

        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.softmax(x)
        return x

# create the model
model = SimpleImageClassifier(2).to(DEVICE)
loss_function = loss_function = nn.CrossEntropyLoss(reduction="mean") # mean is default 
optimizer = optim.AdamW(model.parameters(), lr=1e-3)


# define the train/test loop
# prettier to define as separate methods, but works

# for e epochs
for e in tqdm(range(EPOCHS), desc="Epochs"):

    loss_epoch = {
        "train" : [],
        "val"  : []
    }

    val_acc = []

    # iterate over the train loader
    model.train()
    for i_step, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc="Training"):
        inputs, targets = data
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(inputs)

        loss = loss_function(preds, targets)
        loss_epoch["train"].append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # iterate over val loader and test model
    model.eval() #disables certain functions (such as dropout)
    with torch.no_grad():
        for i_step, data in tqdm(enumerate(val_loader), total=len(val_loader), leave=False, desc="Validation"):
            inputs, targets = data
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(inputs)

            loss = loss_function(preds, targets)
            loss_epoch["val"].append(loss.item())

            # calc some other metrics
            # if you dont want to do it yourself use some other package (e.g. sklearn, torchmetrics)
            val_acc.append((torch.argmax(preds, dim=1) == targets).sum() / targets.size(1))

    print("Epoch:", e)
    print("Training:", "Mean Loss", np.mean(loss_epoch["train"]))
    print("Validation:", "Mean Loss", np.mean(loss_epoch["val"]), "Mean Acc", np.mean(val_acc))
    
# save model weights (load using "model.load_state_dict(torch.load(PATH))" )
# see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save(model.state_dict(), "./my_fancy_model.pkl")
