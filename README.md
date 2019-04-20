# IoT_AnomalyDetection
The project folders structure is as follows:
----args
----Baseline <- argument files for running training of baseline models using deep autoencoders
----cluster <- argument files for running training of cluster models using deep autoencoders
----Lstm <- argument files for running training models using lstm-encoder-decoder
----Lstm_cluster <- argument files for running training of cluster models using lstm-encoder-decoder
----data <- dataset per IoT device, each folder contains benign dataset and attacks
----Danmini_Doorbell
----gafgyt_attacks
----mirai_attacks
----Ecobee_Thermostat
----gafgyt_attacks
----mirai_attacks
----Ennio_Doorbell
----gafgyt_attacks
----Philips_B120N10_Baby_Monitor
----gafgyt_attacks
----mirai_attacks
----Provision_PT_737E_Security_Camera
----gafgyt_attacks
----mirai_attacks
----Provision_PT_838_Security_Camera
----gafgyt_attacks
----mirai_attacks
----Samsung_SNH_1011_N_Webcam
----gafgyt_attacks
----SimpleHome_XCS7_1002_WHT_Security_Camera
----gafgyt_attacks
----mirai_attacks
----SimpleHome_XCS7_1003_WHT_Security_Camera
----gafgyt_attacks
----mirai_attacks
----figures <- learning curves figures are saved here
----models <- trained models are saved here
----notebooks <- ipynb notebooks, basically for data exploration
----src <- main project code
----Config.py <- handling argument files
----ConfigConsts.py <- config constants
----DatasetConsts.py <- common constants of the dataset, such as dataset columns
----IoTAnomalyDetectorAutoEncoder.py <- implementation of AutoEncoder model
----IoTAnomalyDetectorAutoEncoderNet.py<-Implementation of the network of of AutoEncoder using pytorch
----IoTAnomalyDetectorBase.py <- Base class for training and prediction using pytorch
----IoTAnomalyDetectorLSTMEncDec.py <- implementation of LSTM-Encoder-Decoder model
----IoTAnomalyDetectorLSTMEncDecNet.py<-Implementation of the net of of LSTM-Enc-Dec using pytorch
----IoTAnomalyDetectorNetBase.py <- Base class of implementing NN using pytorch
----main.py <- main file
----NBaIoTDatasetLoader.py <- helper class for loading NBaIoT dataset
----utils.py <- common utils
----tracking <- tracking of running experiments
How to run the code:
The main file is main.py, it receives different arguments, use --help to see all options.
In order to reproduce all experiment results you can use the pass as an argument args file with @ prefix. For example, in order to reproduce baseline results of Danmini doorbell you should run:
python ./main.py @../args/Baseline/Danmini_Doorbell_Baseline.txt 