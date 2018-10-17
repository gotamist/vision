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
#        word_idx = 0 #the start-word
        
        output_indices = []
        for i in range(max_len):
            lstm_out, states = self.lstm( inputs, hidden_states )
#            print( "lstm_out: ", lstm_out[0].size()  )
            output = self.hidden2output( lstm_out )
#            print( output[0])
#            print("output: ", output[0].size())
            maxword = output[0].argmax(dim=1)
#            print( "maxindex: ", maxindex)
#            print( len( output[0] ) )
#            output_indices.append( maxword.cpu().numpy()  )
#            word_arr = int( maxword.data[0].cpu().numpy() )
#            print(  word_arr[0] )
            output_indices.append( int( maxword.data[0].cpu().numpy() ) )
            inputs = self.word_embedding( maxword.unsqueeze(0) ) 
            
#        output = [ item[0] for item in output_indices]                
#        output = [ data_loader.dataset.vocab.idx2word(idx) for idx in output_indices ]
        return  output_indices
