import torch
import torch.nn as nn
import cv2
import numpy as np

IMG_SIZE = 128

classes = [
"MildDemented",
"ModerateDemented",
"NonDemented",
"VeryMildDemented"
]

class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(3,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3)

        self.fc1 = nn.Linear(32*30*30,128)
        self.fc2 = nn.Linear(128,4)

    def forward(self,x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.reshape(x.size(0),-1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = CNN()

model.load_state_dict(torch.load("model/alzheimer_cnn.pth"))

model.eval()


image = cv2.imread("test.jpg")

image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

image = image/255.0

image = np.transpose(image,(2,0,1))

image = torch.tensor(image).unsqueeze(0).float()


with torch.no_grad():

    output = model(image)

    _,prediction = torch.max(output,1)


print("Neurological Aging Stage:",classes[prediction.item()])