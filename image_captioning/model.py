import torch
import torch.nn as nn
import torchvision.models as models
import copy
import numpy as np
from torch.nn import functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_ratio=0.2):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size=hidden_size

        # We'll use this to embed each output as the next input
        self.word_embedding = nn.Embedding(vocab_size, embed_size) #should it be vocab_size instead of 1?

        # Define the LSTM  
        self.lstm=nn.LSTM( embed_size, hidden_size, num_layers=num_layers, dropout=drop_ratio, batch_first=True )

        #produce the output
        self.hidden2output=nn.Linear(hidden_size,vocab_size ) #Can produce the index instead of a vector of size vocab_size

        #dropout
        self.drop_layer = nn.Dropout(p=drop_ratio)
        
        #softmax
        self.l_softm = nn.LogSoftmax(dim=-1)
        
        
    def forward(self, features, captions):
        # Define the feedforward behavior of the model
        caption_embeddings = self.word_embedding( captions[:,:-1]) #cutting of the end_word is purely to pass the assert in 1_Preliminaries
        #unsqueeze is to repeat across the same vector across time.  It plays the role of RepeatVector in keras    
        lstm_input = torch.cat( (features.unsqueeze(1), caption_embeddings), dim=1 ) 
        lstm_out, _ = self.lstm( lstm_input )
#        lstm_out = self.drop_layer( lstm_out )
        outputs = self.hidden2output( lstm_out )
        
        return outputs 
        
    def sample(self, inputs, hidden_states=None, max_len=20):
        "This is the simple greedy sampler. It accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output_indices = []
        for i in range(max_len):
            lstm_out, hidden_states = self.lstm( inputs, hidden_states )
            output = self.hidden2output( lstm_out )
            maxword = output[0].argmax(dim=1)
            word_index = int( maxword.data[0].cpu().numpy() )
            output_indices.append( word_index  )
            inputs = self.word_embedding( maxword.unsqueeze(0) ) 
        return  output_indices
    
    def beam_sample(self, inputs, hidden_states=None, max_len=20, beam_width=8, return_best_only=True):
        """Accept a pre-processed image tensor and return the top predicted 
        sentences using a beam search.
        """
        sequence_pack = [ [ inputs, hidden_states, 0, [] ] ] #start with a single input, expant to BW in subsequent steps
        
        for j in range(max_len):
            # list to keep beam_width^2 potential sequences, of which, we will retain the best beam_width ones before moving to the next j
            store = []
            for stub_info in sequence_pack:
                lstm_out, hidden_states = self.lstm( stub_info[0], stub_info[1] )
                output = self.hidden2output( lstm_out.squeeze(1) )
                log_prob = F.log_softmax( output, -1 ) # do not use separate softmax and log - numerical issues (roundoff)
                # now sort and pick the highest beam_width probabilities
                scores, indices = log_prob.topk( beam_width, -1 )
                indices = indices.squeeze(0)
                for i in range(beam_width):
                    extended_stub = copy.copy(stub_info[3]) #the original stub 
                    extended_stub.append( indices[i].item() )
                    new_score = copy.copy(stub_info[2]) 
                    new_score += scores[0][i].item()
                    inputs = self.word_embedding( indices[i].unsqueeze(0).unsqueeze(0) )
                    store.append( [inputs, hidden_states, new_score, extended_stub] )
                #From this store of beam_width^2 candidate extended_stubs, pick the beam_width ones with the highest new_score values
                store.sort( key=lambda x:x[2], reverse=True)
                sequence_pack = store[ :beam_width ]

        sentences_to_return = [  stub_info[-1] for stub_info in sequence_pack ]
        if return_best_only: sentences_to_return = sentences_to_return[0]
        return sentences_to_return                    
            
                
                
                
        