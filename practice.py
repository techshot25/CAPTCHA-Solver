"""Data image recognizer

Source: https://www.kaggle.com/fournierp/captcha-version-2-images

"""
#%% Import modules and set constants

from matplotlib import pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.nn import functional as F
import torch

DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
PATH = 'computer_vision/samples/*.png'

#%% Import all images and labels 

pics = {}
for image_file in glob(PATH):
    label = image_file.split('/')[-1].split('.')[0]
    with Image.open(image_file) as pic:
        pics[label] = np.asarray(pic)


# The data needs to be array-like for train_test_split
x = list(map(transforms.ToTensor(), pics.values()))
y = list(map(list, pics.keys()))

#%% Set up the encoder to create ones and zeros

encoder = OneHotEncoder().fit(y)
y_encoded = encoder.transform(y).toarray()

#%% split into train and test sets

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.33, random_state=42)

#%% tensorize all of them to prep for neural network

x_train = torch.stack(x_train)
x_test = torch.stack(x_test)
y_train = torch.tensor(y_train)

#%% Define the CNN

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(24, 48, kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(5760, 95)

    def forward(self, x):
        x = self.feats(x)
        x = x.flatten(1)
        x = self.fc(F.relu(x))
        return F.softmax(x, -1)

model = CNN()
model.double().to(DEVICE)

error = nn.BCELoss()
opt = optim.Adam(model.parameters())

#%%

def train(x, y, num_epochs=1000):
    losses = []
    model.train()
    for i in range(num_epochs):
        opt.zero_grad()
        out = model(x.double().to(DEVICE))
        loss = error(out, y_train.to(DEVICE))
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i % int(0.1 * num_epochs) == 0:
            print(i, loss.item())
    return losses

losses = train(x_train, y_train)
plt.plot(losses)
plt.title('BCE loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#%%

with torch.no_grad():
    out = model(x_test.double().to(DEVICE)).cpu().detach().numpy()

res = encoder.inverse_transform(out)
true = encoder.inverse_transform(y_test)

test_images = list(map(transforms.ToPILImage(), x_test))

#%%

s = np.random.choice(len(x_test) - 1)
print(f'True: {true[s]}\nPred: {res[s]}')
plt.imshow(test_images[s])
plt.show()

#%%

def multi_label_score(true, res):
    score = []
    for x, y in zip(true, res):
        if all(x == y):
            score.append(1)
        elif any(x == y):
            score.append(sum(x == y) / len(x))
        else:
            score.append(0)
    return sum(score) / len(score) * 100

print(f'Accuracy score: {multi_label_score(true, res):.4g}%')

# %%
