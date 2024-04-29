import torch
import torch.nn as nn
import muspy
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence
from tqdm.auto import tqdm

def rnn_single_step(current_input:torch.Tensor, prev_hidden:torch.Tensor, hh_weight:torch.Tensor, ih_weight:torch.Tensor, bias:torch.Tensor) -> torch.Tensor:
  '''
  This function 
  
  Arguments:
    current_input: Input vector of the current time step. Has a shape of [input_dimension]
    prev_hidden: Hidden state from the previous time step. Has a shape of [hidden_dimension]
    hh_weight: Weight matrix for from hidden state to hidden state. Has a shape of [hidden_dimension, hidden_dimension]
    ih_weight: Weight matrix for from current input to hidden state. Has a shape of [input_dimension, hidden_dimension]
    bias: Bias of RNN. Has a shape of [hidden_dimension]
  
  Outputs:
    current hidden: Updated hidden state for the current time step. Has a shape of [hidden_dimension]
  
  TODO: Complete this function
  '''
  return 


def initialize_hidden_state_for_single_batch(hidden_dim:int) -> torch.Tensor:
  '''
  This function returns zero Tensor for a given hidden dimension. This function assumes that the RNN uses single layer and single direction.
  
  Argument
    hidden_dim
    
  Return
    initial_hidden_state: Has a shape of [hidden_dim]
  
  TODO: Complete this function
  '''
  return 


def rnn_for_entire_timestep(input_seq:torch.Tensor, prev_hidden:torch.Tensor, hh_weight:torch.Tensor, ih_weight:torch.Tensor, bias:torch.Tensor) -> tuple:
  '''
  This function returns the output of RNN for the given 'input_seq', for the given RNN's parameters (hh_weight, ih_weight, and bias)
  
  Arguments:
    input_seq: Sequence of input vector. Has a shape of [number_of_timestep, input_dimension]
    prev_hidden: Hidden state from the previous time step. Has a shape of [hidden_dimension]
    hh_weight: Weight matrix for from hidden state to hidden state. Has a shape of [hidden_dimension, hidden_dimension]
    ih_weight: Weight matrix for from current input to hidden state. Has a shape of [input_dimension, hidden_dimension]

  
  Return: tuple (output, final_hidden_state)
    output (torch.Tensor): Sequence of output hidden state of RNN along input timesteps. Has a a shape of [number_of_timestep, hidden_dimension]
    final_hidden_state (torch.Tensor): Hidden state of RNN of the last time step. Has a a shape of [hidden_dimension]
    
  TODO: Complete this function using your 'rnn_single_step()'
  '''
  
  return

class CustomEmbeddingLayer(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.weight = torch.randn(num_embeddings, embedding_dim)
    
  def forward(self, x:torch.LongTensor):
    '''
    Argument
      x: torch.LongTensor of arbitrary shape, where each element represent categorical index smaller than self.num_embeddings
      
    Return
      out (torch.Tensor): torch.FloatTensor with [shape of x, self.embedding_dim]
    
    TODO: Complete this function using self.weight
    '''
    
    return
  
class MelodyDataset:
  def __init__(self, muspy_dataset, vocabs=None):
    '''
    The dataset takes vocabs as an argument. If vocabs is None, the dataset will automatically generate vocabs from the given dataset.
    This is useful when you want to use the same vocabs for training and test dataset.
    
    '''
    self.dataset = muspy_dataset
    
    if vocabs is None:
      '''
      Even though you don't have to add 'pad' when if you only use PackedSequence, we will add 'pad' to the vocabulary just in case using paddings.
      '''
      self.idx2pitch, self.idx2dur = self._get_vocab_info()
      self.idx2pitch = ['pad', 'start', 'end'] + self.idx2pitch 
      self.idx2dur = ['pad', 'start', 'end'] + self.idx2dur
      self.pitch2idx = {x:i for i, x in enumerate(self.idx2pitch)}
      self.dur2idx = {x:i for i, x in enumerate(self.idx2dur)}
      
    else:
      self.idx2pitch, self.idx2dur, self.pitch2idx, self.dur2idx = vocabs
    
  def _get_vocab_info(self):
    entire_pitch = []
    entire_dur = []
    for note_rep in self.dataset:
      pitch_in_piece = note_rep[:, 1]
      dur_in_piece = note_rep[:, 2]
      entire_pitch += pitch_in_piece.tolist()
      entire_dur += dur_in_piece.tolist()
    return list(set(entire_pitch)), list(set(entire_dur))
  
  def get_vocabs(self):
    return self.idx2pitch, self.idx2dur, self.pitch2idx, self.dur2idx
    
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    '''
    This dataset class returns melody information as a tensor with shape of [num_notes, 2 (pitch, duration)].
    
    To train a melody language model, you have to provide a sequence of original note, and a sequence of next note for given original note.
    In other word, melody[i+1] has to be the shifted_melody[i], so that melody[i]'s next note can be retrieved by shifted_melody[i]
    (Remember, language model is trained to predict the next upcoming word)
    
    Also, to make genration easier, we usually add 'start' token at the beginning of sequence, and 'end' token at the end of the sequence.
    With these tokens, we can make the model recognize where is the start and end of the sequence explicitly.
    
    You have to add these tokens to the note sequence at this step.
    
    Argument:
      idx (int): Index of data sample in the dataset
    
    Returns:
      melody (torch.LongTensor): Sequence of [categorical_index_of_pitch, categorical_index_of_duration]
                                 Has a shape of [1 (start_token) + num_notes, 2 (pitch, dur)]. 
                                 The first element of the sequence has to be the index for 'start' token for both pitch and duration.
                                 The melody should not include 'end' token 
                                 (Because we don't have to predict next note if we know that current note is 'end' token)
      shifted_melody (torch.LongTensor): Sequence of [categorical_index_of_pitch, categorical_index_of_duration]
                                         Has a shape of [num_notes + 1 (end_token), 2 (pitch, dur)]
                                         The i'th note of shifted melody has to be the same with (i+1)'th note of melody
                                         The shifted melody should not include 'start' token 
                                         (Because we never get a 'start' token after a note)

    TODO: Complete this function
    '''
    
    return

def pack_collate(raw_batch:list):
  '''
  This function takes a list of data, and returns two PackedSequences
  
  Argument
    raw_batch: A list of MelodyDataset[idx]. Each item in the list is a tuple of (melody, shifted_melody)
               melody and shifted_melody has a shape of [num_notes (+1 if you don't consider "start" and "end" token as note), 2]
  Returns
    packed_melody (torch.nn.utils.rnn.PackedSequence)
    packed_shifted_melody (torch.nn.utils.rnn.PackedSequence)

  TODO: Complete this function
  '''  
  return 


class MelodyLanguageModel(nn.Module):
  def __init__(self, hidden_size, embed_size, vocabs):
    super().__init__()
    
    self.idx2pitch, self.idx2dur, self.pitch2idx, self.dur2idx = vocabs
    self.hidden_size = hidden_size
    self.embed_size = embed_size
    self.num_pitch = len(self.idx2pitch)
    self.num_dur = len(self.idx2dur)
    self.num_layers = 3
    
    
    '''
    TODO: Declare four modules. Please follow the name strictly.
      1) self.pitch_embedder: nn.Embedding layer that embed pitch category index to a vector with size of 'embed_size'
      2) self.dur_embedder = nn.Embedding layer that embed duration category index to a vector with size of 'embed_size'
      3) self.rnn = nn.GRU layer that takes concatenated_embedding and has a hidden size of 'hidden_size', num_layers of self.num_layers, and batch_first=True
      4) self.final_layer = nn.Linear layer that takes self.rnn's output and convert it to logits (that can be used as input of softmax) of pitch + duration
    '''    
    
  def get_concat_embedding(self, input_seq):
    '''
    This function returns concatenated pitch embedding and duration embedding for a given input seq
    
    Arguments:
      input_seq: A batch of melodies represented as a sequence of vector (pitch_idx, dur_idx). 
                 Has a shape of [num_batch, num_timesteps (num_notes), 2(pitch, dur)], or [num_timesteps (num_notes), 2]
                 벡터 (pitch_idx, dur_idx)의 시퀀스로 표현된 멜로디들의 집합으로 이루어진 배치. 
                 Shape은 [배치 샘플 수, 타임스텝의 수 (==음표의 수), 2 (음고, 길이)] 혹은 [타임스텝의 수 (num_notes), 2]
    Return:
      concat_embedding: A batch of sequence of concatenated embedding of pitch embedding and duration embedding.
                        Has a shape of [num_batch, num_timesteps (num_notes), embedding_size * 2]
                        Each vector of time t is [pitch_embedding ; duration_embedding] (concatenation)
                        
                        pitch embedding is the output of an nn.Embedding layer of given note pitch index
                        duration embedding is the output of an nn.Embedding layer of given note duration index
    
    TODO: Complete this function using self.pitch_embedder and self.dur_embedder
    You can use torch.cat to concatenate two tensors or vectors
    '''
    
    return 
  
  
  def initialize_rnn(self, batch_size: int) -> torch.Tensor :
    '''
    This function returns initial hidden state for self.rnn for given batch_size
    
    Argument
      batch_size (int): 
      
    Return
      initial_hidden_state (torch.Tensor):
    '''
    
    return torch.zeros([self.num_layers, batch_size, self.hidden_size])
  
    
  
  def forward(self, input_seq:torch.LongTensor):
    '''
    Forward propgation of Melody Language Model.
    
    Argument
      input_seq: A batch of melodies represented as a sequence of vector (pitch_idx, dur_idx). 
                 Has a shape of [num_batch, num_timesteps (num_notes), 2(pitch, dur)], or can be a PackedSequence
                 벡터 (pitch_idx, dur_idx)의 시퀀스로 표현된 멜로디들의 집합으로 이루어진 배치. 
                 Shape은 [배치 샘플 수, 타임스텝의 수 (==음표의 수), 2 (음고, 길이)] 혹은 PackedSequence.
    
    Output
      pitch_dist: Probability distribution of pitch of next upcoming note for each timestep 't'.
                  Has a shape of [num_batch, numtimesteps, self.num_pitch]
                매 타임 스텝 t에 대해, 그 다음에 등장할 음표 음고의 확률 분포
      dur_dist: Probability distribution of duration of next upcoming note for each timestep 't'.
                Has a shape of [num_batch, numtimesteps, self.num_dur]
                매 타임 스텝 t에 대해, 그 다음에 등장할 음표 길이의 확률 분포
      
    '''
      
  
    '''
    TODO: Complete this function. You have to handle both cases: input_seq as ordinary Tensor / input_seq as PackedSequence
    If the input_seq is PackedSequence, return PackedSequence
    
    
    input_seq → self.get_concat_embedding → self.rnn → self.final_layer → torch.softmax for [pitch, duration]
    
    Follow the instruction
    '''

    if isinstance(input_seq, torch.Tensor): # If input is an ordinary tensor

      # 1. Get concatenated_embeddings using self.get_concat_embedding
      
      # 2. Put concatenated_embeddings to self.rnn.
      # Remember: RNN, GRU, LSTM returns two outputs
      
      # 3. Put rnn's output with a shape of [num_batch, num_timestep, hidden_size] to self.final_layer
      
      # 4. Convert logits (output of self.final_layer) to pitch probability and duration probability
      # Caution! You have to get separately softmax-ed pitch and duration
      # Because you have to pick one pitch and one duration from the probability distribution

      pass # Delete this after you complete the code
    elif isinstance(input_seq, PackedSequence):      
      # 1. Get concatenated_embeddings using self.get_concat_embedding
      # To get concatenated_embeddings, You have to either pad_packed_sequence(input_seq, batch_first=True)
      # Or use input_seq.data, and then make new PackedSequence using concatenated_embeddings as data, and copy batch_lengths, sorted_indices, unsorted_indices.
      
      # 2. Put concatenated embedding to self.rnn
      
      # 3. Put rnn output to self.final_layer to get probability logit for pitch and duration
      # Again, rnn's output is PackedSequence so you have to handle it
      
      # 4. Convert logits to pitch probability and duration probability
      # Caution! You have to get separately softmax-ed pitch and duration
      # Because you have to pick one pitch and one duration from the probability distribution
      
      # Return output as PackedSequence
      pass # Delete this after you complete the code
    else:
      print(f"Unrecognized input type: {type(input_seq)}")
    
    return

def get_nll_loss(prob_distribution, correct_class):
  '''
  This function takes predicted probability distrubtion and the corresponding correct_class.
  
  For example,  prob_distribution = [[0.2287, 0.2227, 0.5487], [0.1301, 0.4690, 0.4010]] means that
  for 0th data sample, the predicted probability for 0th category is 0.2287, for 1st category is 0.2227, and for 2nd category is 0.5487,
  and for 1st data sample, the predicted probability for 0th category is 0.1301, for 1st category is 0.4690, and for 2nd category is 0.4010,
  
  Negative Log Likelihood, which is -y*log(y_hat), can be regarded as negative log value of predicted probability for correct class (y==1).
  If the given correct_class is [1, 2], the loss for 0th data sample becomes negative log of [0.2287, 0.2227, 0.5487][1], which is -torch.log(0.2227), 
  because the correct category for this sample was 1st category, and the predicted probability was 0.2227
  And the loss for 1st data sample becomes negative log of [0.1301, 0.4690, 0.4010][2], which is -torch.log(0.4010),
  because the correct category for this sample was 2nd category, and the predicted probability was 0.4010
  
  To make implementation easy, let's assume we have 2D tensor for prob_distribution and  1D tensor for correct_class
   
  Arguments:
    prob_distribution (2D Tensor)
    correct_class (1D Tensor)
    
  Return:
    loss (torch.Tensor): Negative log likelihood loss for every data sample in prob_distrubition. Has a same shape with correct_class
  
  TODO: Complete this function
  
  Caution:  Do not return the mean loss. Return loss that has same shape with correct_class
  Try not to use for loop, or torch.nn.CrossEntropyLoss, or torch.nn.NLLLoss
  '''
  assert prob_distribution.dim() == 2 and correct_class.dim() == 1, "Let's assume we only take 2D tensor for prob_distribution and 1D tensor for correct_class"
  # Write your code from here
  
  return

def get_loss_for_single_batch(model, batch, device):
  '''
  This function takes model and batch and calculate Cross Entropy Loss for given batch.
  
  Arguments:
    model (MelodyLanguageModel)
    batch (batch collated by pack_collate): Tuple of (melody_batch, shifted_melody_batch)
    device (str): cuda or cpu. In which device to calculate the batch
    
  Return:
    loss (torch.Tensor): Calculated mean loss for given model and batch
    
  TODO: Complete this function using get_nll_loss().
  Now you have to return the mean loss of every data sample in the batch 
  
  Caution: You have to calculate loss for pitch, and loss for duration separately.
  Then you can take average of pitch_loss and duration_loss
  
  Important Tip: If you are using PackedSequence, you can feed PackedSequence.data directly to get_nll_loss.
  It makes the implementation much easier, because it doesn't need to reshape probabilty distribution to 2D and correct class to 1D.
  '''
  

  return

def get_initial_input_and_hidden_state(model, batch_size=1):
  '''
  This function generates initial input vector and hidden state for model's GRU
  
  To generate a new sequence, you have to provide initial seed token, which is ['start', 'start'].
  You have to make a initial vector that has [pitch_category_index_of_'start', duration_category_index_of_'start']
  
  You also have to initial hidden state for the model's RNN.
  In uni-directional RNN(or GRU), hidden state of RNN has to be a zero tensor with shape of (num_layers, batch_size, hidden_size)

  
  Argument:
    model (MelodyLanguageModel)
    
  Returns:
    initial_input_vec (torch.Tensor): Has a shape of [batch_size, 1 (timestep), 2]
    initial_hidden (torch.Tensor): Has a shape of [num_layers, bach_size, hidden_size]
    
  TODO: Complete this function
  '''
  
  return


def predict_single_step(model, cur_input, prev_hidden):
  '''
  This function runs MelodyLangaugeModel just for one step, for the given current input and previous hidden state.
  
  Arguments:
    model (MelodyLanguageModel)
    cur_input (torch.LongTensor): Input for the current time step. Has a shape of (batch_size=1, 1 (timestep), 2)
    prev_hidden (torch.Tensor): Hidden state of RNN after previous timestep

  Returns:
    cur_output (torch.LongTensor): Sampled note [pitch_category_idx, duration_category_idx] from the predicted probability distribution, with shape of [1,1,2]
    last_hidden (torch.Tensor): Hidden state of RNN
  Think about running the model.forward() step-by-step.
  
  input_seq → self.get_concat_embedding → self.rnn → self.final_layer → torch.softmax for [pitch, duration] → sampled [pitch, duration]

  TODO: Complete this function
  '''
  return 


def is_end_token(model, cur_output):
  '''
  During the generation, there is a possibility that the generated note predicted 'end' token for either pitch or duration.
  (In fact, model can even estimate 'start' token during the generation even though it has very low probability)
  
  Using information among (model.pitch2idx, model.dur2idx, model.idx2pitch, model.idx2dur, model.num_pitch, model.num_dur), check whether 
  
  Arguments:
    model (MelodyLanguageModel)
    cur_output (torch.LongTensor): Assume it has shape of [1,1,2 (pitch_idx, duration_idx)]
  
  Return:
    is_end_token (bool): True if cur_output include category index such as 'start' or 'end',
                          else False.
                          
  TODO: Complete this function
  '''
  
  
  return 


def generate(model, random_seed=0):
  '''
  This function generates a new melody sequence with a given model and random_seed.
  
  Arguments:
    model (MelodyLanguageModel)
    random_seed (int): Language model's inference will always generate different result, because it uses random sampling for the prediction.
                       Therefore, if you want to reproduce the same generation result, you have to fix random_seed.
  
  Returns:
    generated_note_sequence (torch.LongTensor): Has a shape of [num_generated_notes, 2]
  
  TODO: Complete this function using get_initial_input_and_hidden_state(), predict_single_step(), is_end_token()
  
  Hint: You can use while loop
        You have to track the generated single note in a list or somewhere. 
  '''
  
  torch.manual_seed(random_seed) # To reproduce the result, we have to control random sequence
  
  '''
  Write your code from here
  '''

  return


def convert_idx_pred_to_origin(pred:torch.Tensor, idx2pitch:list, idx2dur:list):
  '''
  This function convert neural net's output index to original pitch value (MIDI Pitch) and duration value 
  
  Argument:
    pred: generated output of the model. Has a shape of [num_notes, 2]. 
          0th dimension of each note represents pitch category index 
          and 1st dimension of each note represents duration category index
  
  Return:
    converted_out (torch.Tensor): Has a same shape with 'pred'.
    
  TODO: Complete this function
  '''
    
  return 

def convert_pitch_dur_to_note_representation(pitch_dur:torch.LongTensor):
  '''
  This function takes pitch_dur (shape of [num_notes, 2]) and returns the corresponding note representation (shape of [num_notes, 4])
  In note representation, each note is represented as [start_timestep, pitch, duration, velocity]
  
  Since our generation is monophonic, you can regard start_timestep starts from 0 and accumulate the duration of note.
  You can fix velocity to 64.
  
  
  Arguments:
    pitch_dur: LongTensor of note where each note represented as pitch and duration value
    
  return:
    note_repr: numpy.Array with shape of [num_notes, 4]
               each note has value of [start_timestep, pitch, duration, velocity]

  TODO: Complete this function
  Hint: You can use torch.cumsum() to accumulate the duration.
  To convert torch tensor to numpy, you can use atensor.numpy()
  
  '''
  
  return


def main():
  example_input_size = 3
  example_hidden_size = 6
  example_sequence_length = 20

  torch.manual_seed(0)
  example_weight_for_hidden_to_hidden = torch.randn([example_hidden_size, example_hidden_size])
  example_weight_for_input_to_hidden = torch.randn([example_hidden_size, example_input_size])
  example_bias = torch.randn([example_hidden_size])
  example_input_sequence = torch.randn([example_sequence_length, example_input_size])

  initial_hidden = initialize_hidden_state_for_single_batch(example_hidden_size)
  assert initial_hidden.shape == torch.Size([example_hidden_size])

  single_output = rnn_single_step(example_input_sequence[0], initial_hidden, example_weight_for_hidden_to_hidden, example_weight_for_input_to_hidden, example_bias)
  assert torch.allclose(single_output, torch.Tensor([ 0.2690, -0.9982,  0.9929, -0.9535,  1.0000,  0.0081]), atol=1e-4), 'Your output is not correct. Please check your code.'

  total_output = rnn_for_entire_timestep(example_input_sequence, initial_hidden, example_weight_for_hidden_to_hidden, example_weight_for_input_to_hidden, example_bias)

  assert isinstance(total_output, tuple) and len(total_output)==2, "RNN's output has to be tuple of two tensors"
  assert isinstance(total_output[0], torch.Tensor), 'Hidden states has to be a tensor'
  assert torch.allclose(total_output[0][6], torch.tensor([ 0.8273,  0.5121, -0.5701, -0.9566,  0.9984,  0.5125]), atol= 1e-4), f"Output value is different: {total_output[0][6]}"
  assert torch.allclose(total_output[1], torch.tensor([-0.2121, -0.9892, -0.9953,  0.7993,  1.0000, -0.9995]), atol=1e-4), f"Output value is different: {total_output[1]}"

  custom_embedding_layer = CustomEmbeddingLayer(10, example_input_size)
  random_categorical_input = torch.randint(0,10, [3, 2, 2])
  random_categorical_input, custom_embedding_layer(random_categorical_input)

  your_path = 'essen_folk/'
  essen = muspy.EssenFolkSongDatabase(your_path, download_and_extract=True)
  essen.convert()

  essen_entire = essen.to_pytorch_dataset(representation='note')
  essen_split = essen.to_pytorch_dataset(representation='note', splits=(0.8, 0.1, 0.1), random_state=0)
  entire_set = MelodyDataset(essen_entire)

  train_set = MelodyDataset(essen_split['train'], vocabs=entire_set.get_vocabs())
  valid_set = MelodyDataset(essen_split['validation'], vocabs=entire_set.get_vocabs())
  test_set = MelodyDataset(essen_split['test'], vocabs=entire_set.get_vocabs())

  assert len(train_set[0]) == 2, "You have to return two variables at __getitem__"
  assert train_set[0][0].shape == train_set[0][1].shape, "Shape of Melody and Shifted melody has to be the same"

  assert (train_set[0][0][0] == torch.LongTensor([1, 1])).all(), "You have to add start token at the beginning of melody"
  assert (train_set[0][1][-1] == torch.LongTensor([2, 2])).all(), "You have to add end token at the end of melody"

  assert (train_set[0][0][-1] == torch.LongTensor([15, 29])).all(), "Last part of melody must not include the end token"
  assert (train_set[0][1][0] == torch.LongTensor([27, 19])).all(),  "First part of shifted melody must not include the start token"

  assert (train_set[20][0][1:] == train_set[20][1][:-1]).all(), "Check the melody shift"

  '''
  This cell will make error, because the length of each sample is different to each other
  '''


  hidden_size = 16
  embed_size = 20
      
  model = MelodyLanguageModel(hidden_size, embed_size, entire_set.get_vocabs())

  train_loader = DataLoader(train_set, batch_size=4, collate_fn=pack_collate, shuffle=False)
  batch = next(iter(train_loader))
  melody, shifted_melody = batch
  pitch_out, dur_out = model(melody)

  batch = next(iter(train_loader))
  melody, shifted_melody = batch
  padded_melody, _ = pad_packed_sequence(melody, batch_first=True)

  concat_embedding = model.get_concat_embedding(padded_melody)
  print(f'Your concart_embedding: \n{concat_embedding}')

  assert concat_embedding.shape[:-1] == padded_melody.shape[:-1], "Num_batch and num_timestep of concat_embedding has to be the same with input melody"
  assert concat_embedding.shape[2] == embed_size * 2, "Error in size of embedding dimension"
  assert (concat_embedding[0,0,:] == concat_embedding[1,0,:]).all(), "Error: your embedding vectors for the same input notes are different"

  single_loader = DataLoader(train_set, batch_size=1, shuffle=True)
  single_batch = next(iter(single_loader))
  single_melody, single_shifted_melody = single_batch
  pitch_out, dur_out = model(single_melody)

  assert pitch_out.shape == (1,single_melody.shape[1], model.num_pitch),  \
          f"Error in pitch_out.shape. Expected {1,single_melody.shape[1], model.num_pitch}, but got {pitch_out.shape}"
  assert dur_out.shape == (1,single_melody.shape[1], model.num_dur), \
            f"Error in dur_out.shape. Expected {1,single_melody.shape[1], model.num_dur}, but got {dur_out.shape}"

  assert (0<pitch_out).all() and (pitch_out<1).all() and (0<dur_out).all() and (dur_out<1).all(), \
            "Every output must have a value between 0 and 1 "
  assert (torch.abs(torch.sum(pitch_out, dim=-1)-1)<1e-5).all(), \
            "Sum of probability of every pitch class has to be 1"
  assert (torch.abs(torch.sum(dur_out, dim=-1)-1)<1e-5).all(), \
            "Sum of probability of every duration class has to be 1"


  train_loader = DataLoader(train_set, batch_size=5, collate_fn=pack_collate, shuffle=True)
  batch = next(iter(train_loader))
  melody, shifted_melody = batch
  pitch_out, dur_out = model(melody)

  assert isinstance(pitch_out, type(melody)) and isinstance(dur_out, type(melody)), f"Input of model was {type(melody)} but output is {type(pitch_out)}"

  assert (pitch_out.batch_sizes == melody.batch_sizes).all(), \
            "batch_sizes of input and output has to be the same"
  assert len(pitch_out.data) == len(batch[0].data), "Number of notes in input and output has to be the same"
  assert (torch.abs(torch.sum(pitch_out.data, dim=-1)-1)<1e-5).all(), \
            "Sum of probability of every pitch class has to be 1"
  assert (torch.abs(torch.sum(dur_out.data, dim=-1)-1)<1e-5).all(), \
            "Sum of probability of every duration class has to be 1"  

  torch.manual_seed(0)
  prob_distribution = torch.softmax(torch.randn([10, 3]), dim=-1)
  correct_class = torch.randint(0,3, [10])
  print(f"prob_distribution: \n{prob_distribution}, \n correct_class for each datasample: \n {correct_class.unsqueeze(1)}")

  loss = get_nll_loss(prob_distribution, correct_class)
  print('Loss: ', loss)
  assert (torch.abs(loss-torch.Tensor([1.5020, 0.7572, 0.4797, 0.7693, 0.4563, 0.8718, 0.7973, 1.3412, 1.6403, 0.2423]))<1e-4).all(), "Error in loss value"
  model.cpu()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
  num_iter = 20
  DEV = 'cpu'
  model.to(DEV)
  batch = next(iter(train_loader))
  for i in tqdm(range(num_iter)):
    model.train()
    loss = get_loss_for_single_batch(model, batch, device=DEV)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  batch_size = 2
  input_vec, initial_hidden = get_initial_input_and_hidden_state(model, batch_size=batch_size)
  print(f'input_vec: \n{input_vec} \n initial_hidden: \n {initial_hidden}')

  assert input_vec.ndim == 3
  assert initial_hidden.ndim == 3
  assert input_vec.shape == (batch_size, 1, 2)
  assert initial_hidden.shape == (model.num_layers, batch_size, model.hidden_size)

  input_vec, initial_hidden = get_initial_input_and_hidden_state(model, batch_size=1)
  out_note, last_hidden = predict_single_step(model, input_vec, initial_hidden)
  print(f'out_note: \n{out_note} \n last_hidden: \n {last_hidden}')

  assert out_note.ndim == 3
  assert last_hidden.ndim == 3
  assert out_note.shape == (1,1,2)

  assert len(set([predict_single_step(model, input_vec, initial_hidden)[0] for i in range(5)]))==5, 'Generated output has to be different based on random sampling'

  assert not is_end_token(model, torch.LongTensor([[[10, 7]]])), 'This is not end token'
  assert is_end_token(model, torch.LongTensor([[[2, 40]]])), 'This is end token'
  assert is_end_token(model, torch.LongTensor([[[25, 2]]])),  'This is end token'
  assert is_end_token(model, torch.LongTensor([[[2, 2]]])),  'This is end token'
  
  gen_out = generate(model)
  print(f"gen_out: \n {gen_out}")

  assert isinstance(gen_out, torch.LongTensor), f"output of generate() has to be torch.LongTensor, not {type(gen_out)}"
  assert gen_out.ndim == 2, f"output of generate() has to be 2D tensor, not {gen_out.ndim}D tensor"
  assert gen_out.shape[1] == 2
  
  
  converted_out = convert_idx_pred_to_origin(gen_out, model.idx2pitch, model.idx2dur)
  assert converted_out.shape == gen_out.shape
  print(f"convted_out: \n {converted_out}")

  note_repr = convert_pitch_dur_to_note_representation(converted_out)
  print(f"note_reprnote: \n {note_repr}")
  

if __name__ == '__main__':
  
  main()