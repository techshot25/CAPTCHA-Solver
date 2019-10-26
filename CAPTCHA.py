# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\AppData\Local\Temp'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# ## Test

# %%
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = 'samples/*.png'

# %% [markdown]
# Import all images and labels

# %%
pics = {}
for image_file in glob(PATH):
    #label = image_file.split('/')[-1].split('.')[0] # linux version
    label = image_file.split('\\')[-1].split('.')[0]
    with Image.open(image_file) as pic:
        pics[label] = np.asarray(pic)


# The data needs to be array-like for train_test_split
x = list(map(transforms.ToTensor(), pics.values()))
y = list(map(list, pics.keys()))

# %% [markdown]
# Set up the encoder to create ones and zeros

# %%
encoder = OneHotEncoder().fit(y)
y_encoded = encoder.transform(y).toarray()

# %% [markdown]
# Split into train and test sets

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.33, random_state=42)

# %% [markdown]
# Tensorize all of them to prep for neural network

# %%
x_train = torch.stack(x_train)
x_test = torch.stack(x_test)
y_train = torch.tensor(y_train)

# %% [markdown]
# Define the CNN

# %%
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

# %% [markdown]
# Define the train function and start training.

# %%
y_t = y_train.to(DEVICE)
x_t = x_train.double().to(DEVICE)

def train(x, y, num_epochs=1000):
    losses = []
    model.train()
    for epoch in range(1, 1 + num_epochs):
        opt.zero_grad()
        out = model(x.double().to(DEVICE))
        loss = error(out, y_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if epoch % int(0.1 * num_epochs) == 0:
            print(f'Epoch: {epoch} \t loss: {loss.item()}')
    return losses

losses = train(x_t, y_t)
plt.plot(losses)
plt.title('BCE loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# %% [markdown]
# Run model on test set

# %%
with torch.no_grad():
    out = model(x_test.double().to(DEVICE)).cpu().detach().numpy()

# Decode the ones and zeros matrix
res = encoder.inverse_transform(out)
true = encoder.inverse_transform(y_test)

test_images = list(map(transforms.ToPILImage(), x_test))

# %% [markdown]
# Sample and show results

# %%
for s in np.random.choice(len(x_test) - 1, size=5):
    plt.tit(f'True: {true[s]}\nPred: {res[s]}')
    plt.imshow(test_images[s])
    plt.show()

# %% [markdown]
# Write a scoring algorithm for multilabel classifier
# I chose to go with giving it partial credit for getting some characters correct even though it wouldn't pass a CAPTCHA test.

# %%
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

# %% [markdown]
# This is pretty good for a back of the napkin code that tries to cheat CAPTCHA
