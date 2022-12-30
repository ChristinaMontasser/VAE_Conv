
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import PIL
import numpy as np
import torchvision
from torchvision import transforms
import os
from torch.utils.data import Dataset
import torch.nn as nn
import cv2
import tensorflow as tf
import torchvision.transforms as T
import random 

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

import torch
import torch.optim as optim
if torch.cuda.is_available():
    dev = "cuda:0"
    print("gpu up")
else:
    dev = "cpu"
device = torch.device(dev)

learning_rate = 0.0003
num_epochs = 10
batch_size = 10
in_channels = 3


def to_tensor_and_normalize(imagepil): 
    ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0, std=1), torchvision.transforms.Resize((128, 128))])
    return ChosenTransforms(imagepil)
    

class Objects(Dataset):
    def __init__(self, root_dir):
        super(Objects, self).__init__()
        self.root_dir = root_dir
        self.all_filenames = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.all_filenames)
        
    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        imagepil = PIL.Image.open(os.path.join(self.root_dir, selected_filename)).convert('RGB')
        
        image = to_tensor_and_normalize(imagepil)
        
        return image

dt = Objects("C:/Users/miret/Desktop/VAE/CustomDataset")
train_loader = DataLoader(dataset= dt,       
                          batch_size=batch_size, 
                          shuffle=True,
                          )

class Reshape(nn.Module):
    def _init_(self, *args):
        super()._init_()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def _init_(self, *args):
        super()._init_()

    def forward(self, x):
        return x[:, :, :128, :128]

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.dist_dim = latent_dim
        self.encoder = nn.Sequential(
            #128x128x3
                nn.Conv2d(in_channels, 16, stride=(1, 1), kernel_size=(3, 3), padding=1),#128x128x16
                nn.LeakyReLU(0.01),
                nn.Conv2d(16, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),#64x64x32
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),#32x32x64
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),#32x32x64
                nn.Flatten(),
        )    
        
        self.z_mean = torch.nn.Linear(65536, self.dist_dim)
        self.z_log_var = torch.nn.Linear(65536, self.dist_dim)
        
        self.decoder = nn.Sequential(
                torch.nn.Linear(self.dist_dim, 65536),
                Reshape(-1, 64, 32, 32),
                nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),                
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 16, stride=(2, 2), kernel_size=(3, 3), padding=0),                
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(16, in_channels, stride=(1, 1), kernel_size=(3, 3), padding=0), 
                Trim(),  
                nn.Sigmoid()
                )

    def encoding(self, x, genImgs):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = []
        for i in range(genImgs):
          encoded.append(self.sampling(z_mean, z_log_var)) 
        return encoded
        
    def sampling(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).cuda().float()
        z = z_mean + eps * torch.exp(z_log_var/2.0) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.sampling(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


imgs = next(iter(train_loader))
# Checking the dataset
for images in train_loader:  
    print(images.shape)
    break

model=torch.load('modelVAE.pt')
# model = torch._load_('model.pt')
model.eval()
# print(dir(model))




def show_images():
    genImgs = my_scale.get()
    saved = model.encoding((imgs.cuda().float()), genImgs)
    reImages = []
    for i in range(genImgs):
        reImages.append(model.decoder(saved[i]))
    pickedImage =random.randint(0, batch_size)
    image =imgs[pickedImage]
    image = (image.detach().to(torch.device('cpu')))
    image = np.asarray(image).transpose((2, 1, 0))
    image = Image.fromarray((image*255).astype(np.uint8))

    photo = ImageTk.PhotoImage(image)
    label = Label(master, image = photo)
    label.image = photo
    label.grid(row=2, column=1)
    for i in range(genImgs):
        image = reImages[i][pickedImage]
        image = (image.detach().to(torch.device('cpu')))
        image = np.asarray(image).transpose((2, 1, 0))
        image = Image.fromarray((image*255).astype(np.uint8))

        photo = ImageTk.PhotoImage(image)
        label = Label(master, image = photo)
        label.image = photo
        label.grid(row= i+(i%2)+3, column=(int((i/2)%2))+1)
        #3
        #


master = Tk()

l1=tk.Label(master,text="Scale")
l1.grid(row=1,column=1)
my_scale = tk.Scale(master, from_=0, to=12, orient='horizontal')
my_scale.grid(row=1,column=1)
slider = Button(master, text='Generate', command=show_images)
slider.grid(row = 1, column=2, sticky = E)

# model = VAE()
# optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# res = T.ToPILImage(reImages[0].detach().to(torch.device('cpu')))
# print(res)
# Convert the image tensor to a NumPy array
# image = (reImages[0].detach().to(torch.device('cpu')))
# image = np.asarray(image).transpose((2, 1, 0))
# image = Image.fromarray((image*255).astype(np.uint8))

# photo = ImageTk.PhotoImage(image)
# label = Label(master, image = photo)
# label.image = photo
# label.grid(row= 0)
# value = my_scale.get()
# pickedImage =random.randint(0, batch_size)
# for i in range(value):
#     image = reImages[i][pickedImage]
#     image = (image.detach().to(torch.device('cpu')))
#     image = np.asarray(image).transpose((2, 1, 0))
#     image = Image.fromarray((image*255).astype(np.uint8))

#     photo = ImageTk.PhotoImage(image)
#     label = Label(master, image = photo)
#     label.image = photo
#     label.grid(row= i+3)
    # image = im.resize((256,256))
    # photo = ImageTk.im(image)
    # label = Label(master, image = photo)
    # label.image = photo
    # label.grid(row= index+3)

# print(Image.open("C:/Users/miret/Desktop/VAE/CustomDataset/image1.jpg"))
# images = saved
# for index, im in enumerate(images):
#     image = im.resize((256,256))
#     photo = ImageTk.im(image)
#     label = Label(master, image = photo)
#     label.image = photo
#     label.grid(row= index+3)

master.mainloop()



# import tkinter as tk
# from tkinter import filedialog
# from tkinter import messagebox
# import PIL.Image, PIL.ImageTk

# # Create main window
# window = tk.Tk()
# window.title("Image Viewer")

# # Create frame to hold image grid and slider
# frame = tk.Frame(window)
# frame.pack()

# def display_images(image_filenames, grid_size):
#     # Load and display the images
#     for i, filename in enumerate(image_filenames):
#         # Load image
#         image = Image.open(filename)
#         image = image.resize((200, 200), Image.ANTIALIAS)
#         image = ImageTk.PhotoImage(image)

#         # Display image
#         label = tk.Label(root, image=image)
#         label.image = image  # Keep a reference to prevent garbage collection
#         label.grid(row=i // grid_size, column=i % grid_size)

# def update_grid_size(value):
#     # Update the grid size
#     grid_size = int(value)
#     display_images(image_filenames, grid_size)
# root = tk.Tk()
# root.title("Image Grid")

# # Create slider widget
# slider = ttk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, command=update_grid_size)
# slider.pack()

# # Set initial grid size and display images
# grid_size = 5
# image_filenames = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]
# display_images(image_filenames, grid_size)

# root.mainloop()
