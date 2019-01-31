from IoTAnomalyDetectorBase import IoTAnomalyDetectorBase
from IoTAnomalyDetectorAutoEncoderNet import IoTAnomalyDetectorAutoEncoderNet


class IoTAnomalyDetectorAutoEncoder(IoTAnomalyDetectorBase):
    def __init__(self, data, loss, optimizer, learning_rate, epochs, batch_size, train_val_split=0.8):
        IoTAnomalyDetectorBase.__init__(self, data, IoTAnomalyDetectorAutoEncoderNet(), loss, optimizer, learning_rate,
                                        epochs, batch_size, train_val_split)

    def detect_anomalies(self, x_mat):
        preds = self.predict(x_mat)
