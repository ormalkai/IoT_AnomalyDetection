import IoTAnomalyDetectorModelBase
from torchsummary import summary
from torch import nn, cuda


class IoTAnomalyDetectorAutoEncoder(IoTAnomalyDetectorModelBase):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # (1)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (2)
            nn.ReLU(),  # (3)
            nn.Conv2d(32, 64, kernel_size=2),  # (4)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (5)
            nn.ReLU(),  # (6)
            nn.Conv2d(64, 128, kernel_size=2),  # (7)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (8)
            nn.ReLU(),  # (9)
            View(),  # (10)
            nn.Linear(15488, 500),  # (11)
            nn.ReLU(),  # (12)
            nn.Linear(500, 500),  # (13)
            nn.ReLU(),  # (14)
            nn.Linear(500, 30)  # (15)
        )
        if cuda.is_available():
            self.model.cuda()
        summary(self.model, (1, 96, 96))