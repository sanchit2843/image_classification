import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

#If the data is divided into different folders
#image transformations for train and test data
im_size = 150
train_transforms = transforms.Compose([
                                        transforms.Resize((im_size,im_size)),
                                        transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
                                        transforms.RandomRotation(degrees=10),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(size=299),  # Image net standards
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4302, 0.4575, 0.4539), (0.2361, 0.2347, 0.2432))])
test_transforms = transforms.Compose([
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4302, 0.4575, 0.4539), (0.2361, 0.2347, 0.2432))])

#inverse normalization for image plot

inv_normalize =  transforms.Normalize(
    mean=[-0.4302/0.2361, -0.4575/0.2347, -0.4539/0.2432],
    std=[1/0.2361, 1/0.2347, 1/0.2432]
)


#This is template in case the images are not divided into folders and csv file with names and classes are given
'''
im_size = 150
class datagen(Dataset):
  def __init__(self,direc,labels,transform = None):
    self.dir = direc
    self.labels = labels
    self.transform = transform
  def __len__(self):
    return (len(self.dir))

  def __getitem__(self,idx):
    path = self.dir[idx]
    image = cv2.imread(path)
    if self.transform:
      image = self.transform(image)
    label = self.labels[idx]
    return image,label
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((im_size,im_size)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(45),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
datagen_object = datagen(direc1, label,transform)
train_loader = DataLoader(datagen_object, batch_size=32)
inv_normalize =  transforms.Normalize(mean=[-1, -1, -1],
    std=[2, 2, 2]
)
'''
