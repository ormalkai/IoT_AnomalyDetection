from abc import ABC
import torch


class IoTAnomalyDetectorNetBase(ABC):
    def forward(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save_model_to_file(self, model_filename):
        if model_filename is None:
            return
        print("Saving trained model to file {}".format(model_filename))
        torch.save(self.model.state_dict(), model_filename)

    def load_model_from_file(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename))

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
