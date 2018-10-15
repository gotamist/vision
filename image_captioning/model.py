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

        # self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # Define the LSTM  
        self.lstm=nn.LSTM( embed_size, hidden_size )
        self.hidden2output=nn.Linear( hidden_size, 1 ) #Can produce the index instead of a vector of size vocab_size

        #now, initialize the hidden state
        self.hidden=(torch.zeros( num_layers, 1, self.hidden_size),
                     torch.zeros( num_layers, 1, self.hidden_size))
        
    def forward(self, features, captions):
        pass

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
