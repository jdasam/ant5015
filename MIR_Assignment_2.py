
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union, Callable

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd

from MIR_Assignment_2_predefined import MTATDataset, SpecModel, AudioModel, get_roc_auc, get_tpr_fpr

class OnTheFlyDataset(MTATDataset):
  def __init__(self, dir_path:str, split:str='train', num_max_data:int=4000, sr:int=16000):
    super().__init__(dir_path, split, num_max_data, sr)
    
  def __getitem__(self, idx):
    '''
    __getitem__ returns a corresponding idx-th data sample among the dataset.
    In music-tag dataset, it has to return (audio_sample, label) of idx-th data.
    
    OnTheFlyDataset loads the audio file whenever this __getitem__ function is called.
    In this function, you have to implement these things
    
    1) Get the file path of idx-th data sample (use self.labels['mp3_path'])
    2) Load the audio of that file path
    3) Resample the audio sample into frequency of self.sr (You can use torchaudio.functional.resample)
    4) Return resampled audio sample and the label (tag data) of the data sample
    
    Output
      audio_sample (torch.FloatTensor):  
      label (torch.FloatTensor): A tensor with shape of 50 dimension. Each dimension has value either 0 or 1
                                 If n-th dimension's value is 1, it means n-th tag is True for this data sample
    
    TODO: Complete this function
    '''
    audio_sample = None
    label = None

    
    return audio_sample, label
  
class PreProcessDataset(MTATDataset):
  def __init__(self, dir_path:str, split:str='train', num_max_data:int=4000, sr:int=16000):
    super().__init__(dir_path, split, num_max_data, sr)
    
    self.pre_process_and_save_data()
    
  def pre_process_and_save_data(self):
    '''
    self.pre_process_and_save_data loads every audio sample in the dataset, resample it, and save it into pt file.
    In this function, you have to implement these things
    
    1) For every data sample in the dataset, check whether pre-processed data already exists
      - You can get data sample path by self.labels['mp3_path'].values
      - path of pre-processed data can be in the same directory, but with different suffix.
      - You can make it with Path(mp3_path).with_suffix('.pt')
    2) If it doesn't exist, do follow things
      a) Load audio file 
      b) Resample the audio file with samplerate of self.sr
      c) Get label of this audio file
      d) Save {'audio': audio_tensor, 'label':label_tensor} with torch.save
    
    Output
      None
    
    TODO: Complete this function
    '''
    
    
    
  def __getitem__(self, idx):
    '''
    __getitem__ returns a corresponding idx-th data sample among the dataset.
    In music-tag dataset, it has to return (audio_sample, label) of idx-th data.
    
    PreProcessDataset loads the pre-processed pt file whenever this __getitem__ function is called.
    In this function, you have to implement these things
    
    1) Get the pt file path of idx-th data sample (use self.labels)
    2) Load the pre-procssed data of that file path (use torch.load)
    3) Return the audio sample and the label (tag data) of the data sample

    TODO: Complete this function
    '''
    
    
    return audio_sample, label
  
class OnMemoryDataset(MTATDataset):
  def __init__(self, dir_path:str, split:str='train', num_max_data:int=4000, sr:int=16000):
    super().__init__(dir_path, split, num_max_data, sr)
    
    self.loaded_audios = self.load_audio()
    
  def load_audio(self):
    '''
    In this function, you have to load all the audio file in the dataset, and resample them, 
    and store the data on the memory as a python variable
    
    For each data in the dataset,
      a) Load Audio
      b) Resample it to self.sr
      c) Append it to total_audio_datas
    
    Output:
      total_audio_datas (list): A list of torch.FloatTensor. i-th item of the list corresponds to the audio sample of i-th data
                                Each item is an audio sample in torch.FloatTensor with sampling rate of self.sr 
    '''
    total_audio_datas = []
    
    ### Write your code from here

    
    return total_audio_datas

  def __getitem__(self, idx):
    '''
    __getitem__ returns a corresponding idx-th data sample among the dataset.
    In music-tag dataset, it has to return (audio_sample, label) of idx-th data.
    
    OnMemoryDataset returns the pre-loaded audio data that is aved on self.loaded_audios whenever this __getitem__ function is called.
    In this function, you have to implement these things
    
    1) Load the pre-procssed audio data from self.loaded_audios
    2) Return the audio sample and the label (tag data) of the data sample

    TODO: Complete this function
    '''
    
    return audio_sample, label
  

class YourModel(AudioModel):
  # Cautions: You have to define default values for all the parameters
  def __init__(self, sr=, n_fft=, hop_length=, n_mels=, hidden_size=, num_output=):
    super().__init__(sr, n_fft, hop_length, n_mels, hidden_size, num_output)
    
    # TODO: Implement your own model
  
  def forward(self, x):
    # TODO: Implement your own forward pass
    return 
  
  
def get_conv1d_output_with_linear(atensor, conv1d_linear, kernel_size):
  
  batch_size, in_channels, sequence_length = atensor.shape  
  # TODO: Implement the forward pass
  # Assume stride=1 and padding=0 for simplicity
  # To match with the result of nn.Conv1d, flatten the input tensor without changing dimension order
  
  return

def get_conv2d_output_with_linear(atensor, conv2d_linear, kernel_size):
  
  batch_size, in_channels, height, width = atensor.shape
  # TODO: Implement the forward pass
  # Assume stride=1 and padding=0 for simplicity
  # To match with the result of nn.Conv1d, flatten the input tensor without changing dimension order
  
  return


def get_binary_cross_entropy(pred:torch.Tensor, target:torch.Tensor, eps=1e-8):
  '''
  pred (torch.Tensor): predicted value of a neural network model for a given input (assume that the value is output of sigmoid function)
  target (torch.Tensor): ground-truth label for a given input, given in multi-hot encoding

  output (torch.Tensor): Mean Binary Cross Entropy Loss value of every sample
  '''
  # TODO: Complete this function
  return


def get_precision_and_recall(pred:torch.Tensor, target:torch.Tensor, threshold:float):
  '''
  This function calculates precision and recall of given (prediction, target, threshold)
  
  pred (torch.Tensor): predicted value of a neural network model for a given input 
  target (torch.Tensor): ground-truth label for a given input, given in multi-hot encoding

  output
    precision (torch.Tensor): (Number of true positive)/(Number of total positive predictions)
    recall (torch.Tensor): (Number of true positive)/(Number of total positive ground-truth)
    
  IMPORTANT:
    If there is no positive prediction, precision has to be 1
    If there is no positive ground-truth, recall has to be 1
  
  TODO: Complete this function
  '''
  
  # Write your code here
  precision, recall = None, None

  
  '''
  Be careful for not returning nan because of division by zero
  '''
  assert not (torch.isnan(precision) or torch.isnan(recall))
  return precision, recall

def get_precision_recall_auc(pred:torch.Tensor, target:torch.Tensor, num_grid=500):
  '''
  This function returns PR_AUC value for a given prediction and target.
  Assume pred.shape == target.shape
  
  pred (torch.Tensor): predicted value of a neural network model for a given input 
  target (torch.Tensor): ground-truth label for a given input, given in multi-hot encoding

  output (torch.Tensor): Area Under Curve value for Precision-Recall Curve, using rectangle method
  
  TODO: Complete this function using get_precision_and_recall
  '''
  

  return


def draw_pr_auc_curve(pred:torch.Tensor, target:torch.Tensor, num_grid=500):
  '''
  This function draws PR curve for given prediction and target.
  Assume pred.shape == target.shape
  
  pred (torch.Tensor): predicted value of a neural network model for a given input 
  target (torch.Tensor): ground-truth label for a given input, given in multi-hot encoding
  '''
  pr_curve = []
  for thresh in reversed(torch.linspace(0,1,num_grid)):
    precision, recall = get_precision_and_recall(pred, target, threshold=thresh)
    pr_curve.append((recall, precision))
    
  pr_curve = torch.tensor(pr_curve)
  return pr_curve


def get_f1_score(pred:torch.Tensor, target:torch.Tensor, threshold:float):
  '''
  This function calculates F1 score of given (prediction, target, threshold)
  
  pred (torch.Tensor): predicted value of a neural network model for a given input 
  target (torch.Tensor): ground-truth label for a given input, given in multi-hot encoding

  output
    f1_score (torch.Tensor): 2 * (precision * recall) / (precision + recall)
    
  IMPORTANT:
    If there is no positive prediction, precision has to be 1
    If there is no positive ground-truth, recall has to be 1
  '''
  
  # Write your code here
  return


def find_best_threshold_for_each_class(pred:torch.Tensor, target:torch.Tensor, num_grid=100):
  '''
  This function finds the best threshold for each class to maximize F1 score
  
  pred (torch.Tensor): predicted value of a neural network model for a given input
  target (torch.Tensor): ground-truth label for a given input, given in multi-hot encoding
  
  output
    best_thresholds (torch.Tensor): A tensor of best threshold for each class
  '''
  
  # Write your code here
  return

def get_f1_score_for_each_class(pred, target, best_thresholds):
  '''
  This function calculates F1 score for each class
  
  pred (torch.Tensor): predicted value of a neural network model for a given input
  target (torch.Tensor): ground-truth label for a given input, given in multi-hot encoding
  best_thresholds (torch.Tensor): A tensor of best threshold for each class
  
  output
    f1_scores (torch.Tensor): A tensor of F1 score for each class
  '''
  
  # Write your code here
  return

def collect_every_pred_and_label(model:nn.Module, data_loader:DataLoader, device='cuda'):
  '''
  This function collects every prediction and label of a given model and data_loader
  
  model (nn.Module): A neural network model
  data_loader (DataLoader): A DataLoader object that has test data
  
  output
    every_pred (torch.Tensor): A tensor of every prediction of the model, device has to be 'cpu'
    every_label (torch.Tensor): A tensor of every label of the model, device has to be 'cpu'
  '''
  
  # Write your code here
  return


def get_audio_prediction(audio_path:str, model:nn.Module, best_thresholds:torch.Tensor, target_sr=16000):
  '''
  This function takes an audio path, model, sampling rate, and best_thresholds
  and returns the prediction of the model for the audio file.
  
  audio_path (str): A path of audio file
  model (nn.Module): A neural network model
  best_thresholds (torch.Tensor): A tensor of best threshold for each class
  target_sr (int): Sampling rate of audio file
  
  output
    audio (torch.Tensor): A tensor of audio file
    pred (list of str): A list of tags that are predicted to be True
  
  CAUTION: Do not use external variable to get tag names. Use model.vocab to get tag names
  '''
  
  
  return


if __name__ == '__main__':
  data_dir = Path('MTAT_SMALL/')

  your_model = YourModel(sr=16000, n_fft=1024, hop_length=512, n_mels=48, num_output=50, hidden_size=32)
  ckpt = torch.load('your_model_best.pt')
  weight = ckpt['weight'] if 'weight' in ckpt else ckpt
  your_model.load_state_dict(weight)
  your_model.vocab = ckpt['vocab'] if 'vocab' in ckpt else OnTheFlyDataset(data_dir).vocab
  print(f"Model Loaded")
  
  in_channels = 10
  out_channels = 2
  kernel_size = 4

  dummy_input = torch.randn(5, in_channels, 23)

  conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0)
  output = conv1d(dummy_input)
  print(output.shape)

  conv1d_linear = nn.Linear(in_channels * kernel_size, out_channels)
  conv1d_linear.weight.data = conv1d.weight.data.view(out_channels, -1).clone()
  conv1d_linear.bias.data = conv1d.bias.data.clone()


  linear_output = get_conv1d_output_with_linear(dummy_input, conv1d_linear, kernel_size)
  assert linear_output.shape == output.shape, "Output tensors have different shapes"
  assert torch.allclose(output, linear_output, atol=1e-6), "Output tensors are different"
  
  in_channels = 10
  out_channels = 2
  kernel_size = 4


  dummy_input = torch.randn(5, in_channels, 13, 17)

  conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0)
  output = conv2d(dummy_input)

  conv2d_linear = nn.Linear(in_channels * kernel_size * kernel_size, out_channels)
  conv2d_linear.weight.data = conv2d.weight.data.view(out_channels, -1).clone()
  conv2d_linear.bias.data = conv2d.bias.data.clone()



  linear_output = get_conv2d_output_with_linear(dummy_input, conv2d_linear, kernel_size)
  assert linear_output.shape == output.shape, "Output tensors have different shapes"
  assert torch.allclose(output, linear_output, atol=1e-6), "Output tensors are different"


  pre_calculated_data = torch.load('assignment2_data.pt')
  pre_cal_test_pred = pre_calculated_data['test_pred']
  pre_cal_test_label = pre_calculated_data['test_label']
  
  best_thresholds = find_best_threshold_for_each_class(pre_cal_test_pred, pre_cal_test_label, num_grid=100)

  your_audio_path = data_dir / '2/zephyrus-angelus-11-ave_maria__virgo_serena_josquin_des_prez-0-29.mp3'
  your_model.to('cpu')
  your_model.eval()
  y, pred = get_audio_prediction(your_audio_path, your_model, best_thresholds)

  print(f"pred: {pred}")
  
