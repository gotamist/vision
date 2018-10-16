import torch
import torch.nn as nn
import torchvision.models as models


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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size=hidden_size

        # We'll use this to embed each output as the next input
        self.word_embedding = nn.Embedding(vocab_size, embed_size) #should it be vocab_size instead of 1?

        # Define the LSTM  
        self.lstm=nn.LSTM( embed_size, hidden_size, num_layers=num_layers, batch_first=True )

        #produce the output
        self.hidden2output=nn.Linear(hidden_size,vocab_size ) #Can produce the index instead of a vector of size vocab_size
        
#        self.hidden = (torch.zeros( num_layers, batch_size, self.hidden_size),
#                          torch.zeros( num_layers, batch_size, self.hidden_size)  )

        #Function to initialize the hidden state from forward()
#    def init_hidden(num_layers, batch-size):
#            return ( torch.zeros( num_layers, batch_size, self.hidden_size),
#                          torch.zeros( num_layers, batch_size, self.hidden_size) )
        
    def forward(self, features, captions):
        # Define the feedforward behavior of the model
        # Note, start_word has index 0 and end_word is 1
#        self.hidden = init_hidden(num_layers, batch_size)
        caption_embeddings = self.word_embedding( captions[:,:-1])
        lstm_input = torch.cat( (features.unsqueeze(1), caption_embeddings), dim=1 )
       
        lstm_out, _ = self.lstm( lstm_input )  
        outputs = self.hidden2output( lstm_out )
        
        return outputs 
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
