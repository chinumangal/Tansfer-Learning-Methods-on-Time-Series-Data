# Classes for all implemented models
#
#
import torch.nn as nn


"""
    Basic model:
        -first CNN implementation
        -reference for other models
        - 8 Conv Layers, 2 Dense Layers
        -input_shape (1,500)
"""
#Model class
#
#
class Base_Model(nn.Module):

    def __init__(self, input_shape):
        super(Base_Model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()
        
        # Calculate the flattened size
        self.flatten_size = 256 * (input_shape[1])  # 256 channels * 500 timesteps
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc_relu1 = nn.ReLU()

        self.fc2 = nn.Linear(1024, input_shape[0] * input_shape[1])




    def forward(self, x, input_shape):
        x = self.conv1(x)
        x = self.relu1(x)
        #print(f'After conv1: {x.shape}')
        
        x = self.conv2(x)
        x = self.relu2(x)
        #print(f'After conv2: {x.shape}')
        
        x = self.conv3(x)
        x = self.relu3(x)
        #print(f'After conv3: {x.shape}')
        
        x = self.conv4(x)
        x = self.relu4(x)
        #print(f'After conv4: {x.shape}')

        x = self.conv5(x)
        x = self.relu5(x)
        #print(f'After conv5: {x.shape}')
        
        x = self.conv6(x)
        x = self.relu6(x)
        #print(f'After conv6: {x.shape}')

        x = self.conv7(x)
        x = self.relu7(x)
        #print(f'After conv7: {x.shape}')

        x = self.conv8(x)
        x = self.relu8(x)
        #print(f'After conv8: {x.shape}')

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(f'After flattening: {x.shape}')

        x = self.fc1(x)
        x = self.fc_relu1(x)

        x = self.fc2(x)
        x = x.view(x.size(0), input_shape[0], input_shape[1])
        return x

