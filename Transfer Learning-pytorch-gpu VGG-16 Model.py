#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch')


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# # Set Random Seeds from Reproducibility

# In[3]:


torch.manual_seed(42)


# # Check For GPU

# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (f"Using device : {device}")


# In[5]:


df = pd.read_csv("fmnist_small.csv")


# In[6]:


df.head()


# In[7]:


df.shape


# # Create a 4x4 grid of Image

# In[8]:


fig, axes = plt.subplots(4,4, figsize=(10,10))
fig.suptitle("First 16 Images", fontsize = 16)



for i, ax in enumerate(axes.flat):
    img = df.iloc[i,1:].values.reshape(28,28)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Lable:{df.iloc[i,0]}")
    
    
plt.tight_layout(rect=[0, 0, 0.96, 1])  
plt.show()


# # Train test split

# In[9]:


x = df.iloc[:,1:].values
y = df.iloc[:, 0].values


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# # Transformations

# In[11]:


get_ipython().system('pip install torchvision')


# In[12]:


from torchvision.transforms import transforms


custom_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                      
])


# In[13]:


from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    
    def __init__(self, features, labels, transform):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        
        image = self.features[index].reshape(28,28)                          #resize to (28,28)
        
        image = image.astype(np.uint8)                                       #change datatype to np.uint8
        
        image = np.stack([image] * 3, axis=-1)                               #change black & white to color
        
        image = Image.fromarray(image)                                        #convert array to PIL image
        
        image = self.transform(image)                                        #apply transforms
        
        return image, torch.tensor(self.labels[index], dtype = torch.long)    #retuen
        
    


# In[14]:


train_dataset = CustomDataset (x_train, y_train, transform = custom_transform)


# In[15]:


test_dataset = CustomDataset (x_test, y_test, transform = custom_transform)


# In[16]:


train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, pin_memory = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True, pin_memory = True)


# # Fetch the Pretrained Model

# In[17]:


import torchvision.models as models

vgg16 = models.vgg16(pretrained = True)


# In[18]:


vgg16


# In[19]:


vgg16.features


# In[20]:


vgg16.classifier


# In[21]:


for param in vgg16.features.parameters():
    param.requires_grad = False


# In[22]:


vgg16.classifier = nn.Sequential(
    nn.Linear(25088,1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024,512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512,10)

)


# In[23]:


vgg16.classifier


# In[24]:


vgg16 = vgg16.to(device)


# In[25]:


learning_rate = 0.0001
epochs = 20


# In[26]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=learning_rate)


# # Training Loop

# In[27]:


for epoch in range(epochs):
    
    total_epoch_loss = 0
    
    for batch_features, batch_labels in train_loader:
        
        # move data to gpu
        batch_features, batch_lables = batch_features.to(device), batch_labels.to(device)
        
        # forward pass
        outputs = vgg16(batch_features) 
        
        
        # calculate loss
        loss = criterion(outputs, batch_labels)
        
        # back pass
        optimizer.zero_grad()
        loss.backward()
        
        # update grads
        optimizer.step()
        
        total_epoch_loss = total_epoch_loss + loss.item()
       
    
    avg_loss = total_epoch_loss/len(train_loader)
    print(f'Epoch:{epoch + 1}, Loss:{avg_loss}')


# In[28]:


vgg16.eval()


# # Evaluation on Test Data 

# In[29]:


total = 0
correct = 0

with torch.no_grad():
    
    for batch_features, batch_labels in test_loader:
        
        # move data to gpu
        
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        outputs = vgg16(batch_features)
        
        _, predicted = torch.max(outputs,1)
        
        total = total+ batch_labels.shape[0]
        
        correct = correct + (predicted == batch_labels).sum().item()
        

print (correct/total)


# # Evaluation on Training Data 

# In[30]:


total = 0
correct = 0

with torch.no_grad():
    
    for batch_features, batch_labels in train_loader:
        
        # move data to gpu
        
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        outputs = vgg16(batch_features)
        
        _, predicted = torch.max(outputs,1)
        
        total = total+ batch_labels.shape[0]
        
        correct = correct + (predicted == batch_labels).sum().item()
        

print (correct/total)


# In[ ]:




