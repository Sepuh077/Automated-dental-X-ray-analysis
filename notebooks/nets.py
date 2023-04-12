import torch


class SingleToothNet(torch.nn.Module):
    def __init__(self):
        super(SingleToothNet, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(4, 8, kernel_size=3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(8, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.2),

            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(32, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d( (2, 2) ),

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
            # torch.nn.ReLU(),
        )

        self.d1 = torch.nn.Linear(2050, 50)
        # self.d2 = torch.nn.Linear(100, 100)
        # self.d3 = torch.nn.Linear(1000, 200)
        # self.d4 = torch.nn.Linear(200, 100)
        self.d5 = torch.nn.Linear(50, 32)

        self.d6 = torch.nn.Linear(32, 32)

        self.dropout1 = torch.nn.Dropout(p=0.2)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        images, positions = x
        x = images[:, None]
        
        x = self.convs(x)

        # print(x.shape)
        
        x = torch.cat( (positions, x), axis=-1 )

        x = self.tanh( self.d1(x) )

        # x = x + self.tanh( self.d2(x) )

        # x = self.dropout1(x)

        # x = self.tanh( self.d2(x) )
        # x = self.tanh( self.d3(x) )
        # x = self.tanh( self.d4(x) )

        x = self.d5(x)

        # x = self.softmax(x)

        # x = self.pos( positions ) * 0.5 + x * 0.5

        # x = self.tanh(x)

        # x = self.d6(x)

        x = self.softmax(x)
        # x = torch.abs(x)

        return x


class SingleToothNet2(torch.nn.Module):
    def __init__(self):
        super(SingleToothNet2, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(3,  6, kernel_size=3),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(6, 9, kernel_size=3),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.2),

            torch.nn.Conv2d(9, 12, kernel_size=3),
            torch.nn.BatchNorm2d(12),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d( (3, 3) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Flatten(),
        )

        self.d1 = torch.nn.Linear(8280, 32)
        self.d2 = torch.nn.Linear(32, 32)
        # self.d3 = torch.nn.Linear(1000, 200)
        # self.d4 = torch.nn.Linear(200, 50)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.convs(x[:, None])

        x = self.relu( self.d1(x) )

        # x = x + self.relu(x)

        x = self.softmax( self.d2(x) )

        return x
    

class FullTeethModel(torch.nn.Module):
    def __init__(self):
        super(FullTeethModel, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (3, 3) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(4,  12, kernel_size=3),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (3, 3) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(12, 36, kernel_size=3),
            torch.nn.BatchNorm2d(36),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( (2, 2) ),
            # torch.nn.Dropout2d(p=0.2),

            torch.nn.Conv2d(36, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d( (3, 3) ),
            # torch.nn.Dropout2d(p=0.1),

            torch.nn.Flatten(),
        )

        self.d1 = torch.nn.Linear(94464, 200)
        self.d2 = torch.nn.Linear(200, 160)
        # self.d3 = torch.nn.Linear(1000, 200)
        # self.d4 = torch.nn.Linear(200, 50)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.convs(x[:, None])

        x = self.relu( self.d1(x) )

        # x = x + self.relu(x)

        x = self.softmax( self.d2(x) )

        return x
