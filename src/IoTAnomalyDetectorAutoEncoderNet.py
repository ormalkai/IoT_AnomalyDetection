from IoTAnomalyDetectorNetBase import IoTAnomalyDetectorNetBase
from torchsummary import summary
from torch import nn, cuda


class IoTAnomalyDetectorAutoEncoderNet(IoTAnomalyDetectorNetBase):
    def __init__(self):
        self.model = nn.Sequential(
            # Encoder 115 -> 86 -> 57 -> 38 -> 23
            nn.Linear(115, 86),
            nn.Sigmoid(),
            nn.Linear(86, 57),
            nn.Sigmoid(),
            nn.Linear(57, 38),
            nn.Sigmoid(),
            nn.Linear(38, 23),
            nn.Sigmoid(),
            # Decoder 23 -> 38 -> 57 -> 86 -> 115
            nn.Linear(23, 38),
            nn.Sigmoid(),
            nn.Linear(38, 57),
            nn.Sigmoid(),
            nn.Linear(57, 86),
            nn.Sigmoid(),
            nn.Linear(86, 115),
            nn.Sigmoid(),
        )
        if cuda.is_available():
            self.model.cuda()
        summary(self.model, (1, 115))