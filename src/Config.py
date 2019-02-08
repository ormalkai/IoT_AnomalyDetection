from ConfigConsts import *
import argparse


class Config:
    pass


def parse_args(config, args=None):
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--name", default="Experiment", type=str,
                        help="Name of current experiment")
    parser.add_argument("--model", default=MODEL_AUTOENCODER, choices=[MODEL_AUTOENCODER, MODEL_LSTM_ENC_DEC],type=str,
                        help="Which model to use")
    parser.add_argument("--loss", default=LOSS_MSE, choices=[LOSS_MSE], type=str, help="Which loss to use")
    parser.add_argument("--optimizer", default=OPTIM_SGD, choices=[OPTIM_SGD, OPTIM_ADAM], type=str,
                        help="Which optimizer to use")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=250, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--seq_len", default=1, type=int, help="Sequence length for lstm model")
    parser.add_argument("--iots", required=True, action="append",
                        choices=[DANMINI_DOORBELL,
                                 ENNIO_DOORBELL,
                                 ECOBEE_THERMOSTAT,
                                 PHILIPS_BABY_MONITOR,
                                 PROV_737_SECURITY_CAM,
                                 PROV_838_SECURITY_CAM,
                                 SAMSUNG_WEB_CAM,
                                 SIMPLE_HOME_1002_SECURITY_CAM,
                                 SIMPLE_HOME_1003_SECURITY_CAM], type=str, help="Which data to use")
    parser.add_argument("--train_test_split", default=0.66, type=float, help="Train_test split, train ratio")
    parser.add_argument("--train_val_split", default=0.8, type=float, help="Train_val split, train ratio")
    parser.add_argument("--model_filename", default=None, type=str, help="Model file name to save or load")
    parser.add_argument("--train", action="store_true", help="run training")
    parser.add_argument("--test", action="store_true", help="run test")

    parser.parse_args(args=args, namespace=config)
    config.__dict__['is_cli'] = None is args
    print_config(config)


def print_config(config):
    attrs = vars(config)
    print("Configuration:\n"
          "================")
    for key, value in attrs.items():
        print("{}={}".format(key, value))

