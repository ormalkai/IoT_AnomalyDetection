# IoT_AnomalyDetection
## The project folders structure is as follows:<br>
----args<br>
    ----Baseline <- argument files for running training of baseline models using deep autoencoders<br>
    ----cluster <- argument files for running training of cluster models using deep autoencoders<br>
    ----Lstm <- argument files for running training models using lstm-encoder-decoder<br>
    ----Lstm_cluster <- argument files for running training of cluster models using lstm-encoder-decoder<br>
----data <- dataset per IoT device, each folder contains benign dataset and attacks<br>
    ----Danmini_Doorbell<br>
        ----gafgyt_attacks<br>
        ----mirai_attacks<br>
    ----Ecobee_Thermostat<br>
        ----gafgyt_attacks<br>
        ----mirai_attacks<br>
    ----Ennio_Doorbell<br>
        ----gafgyt_attacks<br>
    ----Philips_B120N10_Baby_Monitor<br>
        ----gafgyt_attacks<br>
        ----mirai_attacks<br>
    ----Provision_PT_737E_Security_Camera<br>
        ----gafgyt_attacks<br>
        ----mirai_attacks<br>
    ----Provision_PT_838_Security_Camera<br>
        ----gafgyt_attacks<br>
        ----mirai_attacks<br>
    ----Samsung_SNH_1011_N_Webcam<br>
        ----gafgyt_attacks<br>
    ----SimpleHome_XCS7_1002_WHT_Security_Camera<br>
        ----gafgyt_attacks<br>
        ----mirai_attacks<br>
    ----SimpleHome_XCS7_1003_WHT_Security_Camera<br>
        ----gafgyt_attacks<br>
        ----mirai_attacks<br>
----figures <- learning curves figures are saved here<br>
----models <- trained models are saved here<br>
----notebooks <- ipynb notebooks, basically for data exploration<br>
----src <- main project code<br>
    ----Config.py <- handling argument files<br>
    ----ConfigConsts.py <- config constants<br>
    ----DatasetConsts.py <- common constants of the dataset, such as dataset columns<br>
    ----IoTAnomalyDetectorAutoEncoder.py <- implementation of AutoEncoder model<br>
    ----IoTAnomalyDetectorAutoEncoderNet.py<-Implementation of the network of of AutoEncoder using pytorch<br>
    ----IoTAnomalyDetectorBase.py <- Base class for training and prediction using pytorch<br>
    ----IoTAnomalyDetectorLSTMEncDec.py <- implementation of LSTM-Encoder-Decoder model<br>
    ----IoTAnomalyDetectorLSTMEncDecNet.py<-Implementation of the net of of LSTM-Enc-Dec using pytorch<br>
    ----IoTAnomalyDetectorNetBase.py <- Base class of implementing NN using pytorch<br>
    ----main.py <- main file<br>
    ----NBaIoTDatasetLoader.py <- helper class for loading NBaIoT dataset<br>
    ----utils.py <- common utils<br>
    ----tracking <- tracking of running experiments<br>
## How to run the code:<br>
The main file is main.py, it receives different arguments, use --help to see all options.<br>
In order to reproduce all experiment results you can use the pass as an argument args file with @ prefix. For example, in order to reproduce baseline results of Danmini doorbell you should run:<br>
python ./main.py @../args/Baseline/Danmini_Doorbell_Baseline.txt <br>
