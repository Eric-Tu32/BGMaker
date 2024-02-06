import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
# input_size = 784 # 28x28

#input_size = 88
#hidden_size = 128
#num_layers = 5

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMVAE, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bridge = nn.Sequential(
            Reshape((-1, 1, 1920, hidden_size)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Reshape((-1, 1920, hidden_size))
        )
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        z, _ = self.encoder(x, (h0,c0)) # z: (b, 1920, 128)
        
        z = self.bridge(z)
        
        h0 = torch.zeros(self.num_layers, z.size(0), self.input_size).to(device)
        c0 = torch.zeros(self.num_layers, z.size(0), self.input_size).to(device)
        out, _ = self.decoder(z, (h0,c0))

        return out

input_size = 88
num_layers = 3

class StackedLSTMVAE(nn.Module):
    def __init__(self, input_size, num_layers):
        super(StackedLSTMVAE, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
    
        self.encoder1 = nn.LSTM(input_size, 128, num_layers, batch_first=True)
        self.bn1 = nn.Sequential(
                Reshape((-1, 1, 1920, 128)),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                Reshape((-1, 1920, 128))
        )

        self.encoder2 = nn.LSTM(128, 256, num_layers, batch_first=True)
        self.bn2 = nn.Sequential(
                Reshape((-1, 1, 1920, 256)),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                Reshape((-1, 1920, 256))
        )

        self.decoder2 = nn.LSTM(256, 128, num_layers, batch_first=True)
        self.bn3 = nn.Sequential(
                Reshape((-1, 1, 1920, 128)),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                Reshape((-1, 1920, 128))
        )


        self.decoder1 = nn.LSTM(128, 88, num_layers, batch_first=True)
     
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), 128).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), 128).to(device)
        z1, _ = self.encoder1(x, (h0,c0)) # z: (b, 1920, 128)
        z1 = self.bn1(z1)

        h0 = torch.zeros(self.num_layers, z1.size(0), 256).to(device)
        c0 = torch.zeros(self.num_layers, z1.size(0), 256).to(device)
        z2, _ = self.encoder2(z1, (h0, c0))
        z2 = self.bn2(z2)
       
        h0 = torch.zeros(self.num_layers, z2.size(0), 128).to(device)
        c0 = torch.zeros(self.num_layers, z2.size(0), 128).to(device)
        z3, _ = self.decoder2(z2, (h0, c0))
        z3 = self.bn3(z3)
        
        h0 = torch.zeros(self.num_layers, z3.size(0), 88).to(device)
        c0 = torch.zeros(self.num_layers, z3.size(0), 88).to(device)
        out, _ = self.decoder1(z3, (h0,c0))

        return out

def get_model(input_size=input_size, num_layers=num_layers):
    return StackedLSTMVAE(input_size, num_layers)


if __name__ == "__main__":
    model = StackedLSTMVAE(input_size, num_layers).to(device)
    print("Num Params: ", sum(p.numel() for p in model.parameters()))
    x = torch.randn(32, 1920, 88).to(device)
    x2 = torch.randn(32, 800, 88).to(device)
    z = model(x)
    print(model(x2).shape)
