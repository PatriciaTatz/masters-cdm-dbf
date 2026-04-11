%Start up model and simulink model 

%Initiate the mat file containing required input parameters for the model
load('./mat/cdm.mat');

%Load simulink model 
open_system("./simulink/bpsk_cdm_carrier_lpf_waveform.slx");