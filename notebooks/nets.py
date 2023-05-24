import torch


class SingleToothNet(torch.nn.Module):
    def __init__(self):
        super(SingleToothNet, self).__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),

            torch.nn.Conv2d(4, 8, kernel_size=3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),

            torch.nn.Conv2d(8, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),

            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),

            torch.nn.Conv2d(32, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
        )
        self.pos = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 32),
        )
        self.d1 = torch.nn.Linear(2048, 100)
        self.d2 = torch.nn.Linear(100, 32)
        self.d6 = torch.nn.Linear(32, 32)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        images, positions = x
        x = images[:, None]
        x = self.convs(x)
        x = self.tanh( self.d1(x) )
        x = self.d2(x)
        pos_x = self.pos(positions)
        x = 0.8 * pos_x + 0.2 * x
        x = self.tanh(x)
        x = self.d6(x)
        x = self.softmax(x)
        return x


class SingleToothNet2(torch.nn.Module):
    def __init__(self):
        super(SingleToothNet2, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),

            torch.nn.Conv2d(3,  6, kernel_size=3),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),

            torch.nn.Conv2d(6, 9, kernel_size=3),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),

            torch.nn.Conv2d(9, 12, kernel_size=3),
            torch.nn.BatchNorm2d(12),

            torch.nn.Flatten(),
        )

        self.d1 = torch.nn.Linear(8280, 32)
        self.d2 = torch.nn.Linear(32, 32)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.convs(x[:, None])

        x = self.relu( self.d1(x) )

        x = self.softmax( self.d2(x) )

        return x
    

class FF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = torch.nn.Linear(15000, 200)
        self.d2 = torch.nn.Linear(200, 50)
        self.d3 = torch.nn.Linear(50, 10)

        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)

        x = self.tanh( self.d1(x) )

        x = self.tanh( self.d2(x) )

        x = self.d3(x)

        return x.reshape(-1, 5, 2)


class FF_ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = torch.nn.Linear(15000, 200)
        self.d1_res = torch.nn.Linear(200, 200)
        self.d2 = torch.nn.Linear(200, 50)
        self.d3 = torch.nn.Linear(50, 10)
        self.d3_res = torch.nn.Linear(10, 10)

        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)

        x = self.tanh( self.d1(x) )
        x = x + self.tanh( self.d1_res(x) )

        x = self.tanh( self.d2(x) )

        x = self.tanh( self.d3(x) )
        x = x + self.tanh( self.d3_res(x) )

        return x.reshape(-1, 5, 2)


class Conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.Flatten()
        )
        self.d1 = torch.nn.Linear(1280, 500)
        self.d2 = torch.nn.Linear(500, 200)
        self.d3 = torch.nn.Linear(200, 50)
        self.d4 = torch.nn.Linear(50, 10)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.conv(x[:, None])

        x = self.tanh( self.d1(x) )

        x = self.tanh( self.d2(x) )

        x = self.tanh( self.d3(x) )

        x = self.d4(x)

        return x.reshape(-1, 5, 2)
    

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.Flatten()
        )
        self.d1 = torch.nn.Linear(1280, 500)
        self.d2 = torch.nn.Linear(500, 200)
        self.d2_res = torch.nn.Linear(200, 200)
        self.d3 = torch.nn.Linear(200, 50)
        self.d4 = torch.nn.Linear(50, 10)
        self.d4_res = torch.nn.Linear(10, 10)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.conv(x[:, None])

        x = self.tanh( self.d1(x) )

        x = self.tanh( self.d2(x) )
        x = x + self.tanh( self.d2_res(x) )

        x = self.tanh( self.d3(x) )
        
        x = self.tanh( self.d4(x) )
        x = x + self.tanh( self.d4_res(x) )

        return x.reshape(-1, 5, 2)
