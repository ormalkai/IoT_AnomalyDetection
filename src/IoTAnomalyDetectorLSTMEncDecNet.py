from IoTAnomalyDetectorNetBase import IoTAnomalyDetectorNetBase
from torchsummary import summary
from torch import nn, cuda, random
import torch
import numpy as np


class IoTAnomalyDetectorLSTMEncDecNetImp(nn.Module):
    def __init__(self, encoder_input_size, encoder_hidden_size, decoder_hidden_size, teacher_forcing_ratio):
        super().__init__()
        self.encoder = nn.LSTM(input_size=encoder_input_size, hidden_size=encoder_hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=encoder_input_size, hidden_size=decoder_hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=decoder_hidden_size, out_features=encoder_input_size)
        self.sigmoid = nn.Sigmoid()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder_input_size = encoder_input_size

    def forward(self, x):
        batch_size = x.shape[0]
        max_len = x.shape[1]
        # feed encoder with the whole sequence
        _, (encoder_last_hidden_state, encoder_last_cell_state) = self.encoder(x)

        # reconstruct last output using encoder_last_hidden_state
        predicted_outputs = torch.zeros(batch_size, max_len, self.encoder_input_size)
        predicted_outputs[:, -1, :] = self.sigmoid(self.linear(encoder_last_hidden_state))
        # we use encoder_last_hidden_state, encoder_last_cell_state as the initializer
        # to the the decoder too
        decoder_hidden_state, decoder_cell_state = encoder_last_hidden_state.clone(), encoder_last_cell_state.clone()
        for t in range(max_len - 2, -1, -1):
            decoder_input = predicted_outputs[:, t + 1, :].data
            if self.training and self.teacher_forcing_ratio:
                if np.random.random() < self.teacher_forcing_ratio:
                    decoder_input = x[:, t + 1, :]
            # Add "sequence" dimension since we iterate through lstm one by one
            decoder_input = decoder_input.unsqueeze(1)
            decoder_output, (decoder_hidden_state, decoder_cell_state) = \
                self.decoder(decoder_input, (decoder_hidden_state, decoder_cell_state))
            decoder_output = decoder_output.squeeze(1)
            predicted_outputs[:, t, :] = self.sigmoid(self.linear(decoder_output))
        return predicted_outputs


class IoTAnomalyDetectorLSTMEncDecNet(IoTAnomalyDetectorNetBase):
    def __init__(self, encoder_input_size=115, encoder_hidden_size=256, decoder_hidden_size=256,
                 teacher_forcing_ratio=0.8):
        self.model = IoTAnomalyDetectorLSTMEncDecNetImp(encoder_input_size, encoder_hidden_size, decoder_hidden_size,
                                                        teacher_forcing_ratio)
        if cuda.is_available():
            self.model.cuda()
        # summary(self.model, (100, 115))
