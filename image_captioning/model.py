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
        
        
    def forward(self, features, captions):
        # Define the feedforward behavior of the model
        # Note, start_word has index 0 and end_word is 1
#        self.hidden = init_hidden(num_layers, batch_size)
        caption_embeddings = self.word_embedding( captions[:,:-1]) #cutting of the end_word is purely to pass the assert in 1_Preliminaries
        #unsqueeze is to repeat across the same vector across time.  It plays the role of RepeatVector in keras    
        lstm_input = torch.cat( (features.unsqueeze(1), caption_embeddings), dim=1 ) 
       
        lstm_out, _ = self.lstm( lstm_input )  
        outputs = self.hidden2output( lstm_out )
        
        return outputs 
        
    def sample(self, inputs, hidden_states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output_indices = []
        for i in range(max_len):
            lstm_out, hidden_states = self.lstm( inputs, hidden_states )
            output = self.hidden2output( lstm_out )
#            maxword = output[0].argmax(dim=1)
            maxword = output.argmax(dim=1)
            word_index = int( maxword.data[0].cpu().numpy() )
            output_indices.append( word_index  )
            inputs = self.word_embedding( maxword.unsqueeze(0) ) 
        return  output_indices
    
#        sample_ids = []
#        for i in range(max_len):
#            lstm_features, hidden_states = self.lstm(inputs, hidden_states)
#            lstm_features = self.hidden2output(lstm_features.squeeze(1))
#            predicted = lstm_features.max(1)[1]   
#            sample_ids.append(predicted.item())
#            inputs = self.word_embedding(predicted).unsqueeze(1)             
#        return sample_ids
