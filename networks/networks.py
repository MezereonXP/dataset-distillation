import torch.nn as nn
import torch.nn.functional as F
import logging
import itertools
from . import utils
import math
import torch
#from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel


class LeNet(utils.ReparamModule):
    supported_dims = {28, 32}

    def __init__(self, state):
        if state.dropout:
            raise ValueError("LeNet doesn't support dropout")
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out

class TextConvNet(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        if state.dropout:
            raise ValueError("TextConvNet doesn't support dropout")
        super(TextConvNet, self).__init__()
        #if state.textdata:
        ninp=state.ninp #Maybe 32
        ntoken=state.ntoken
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.encoder.weight.requires_grad = False 
        self.conv1 = nn.Conv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13968, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if state.num_classes <= 2 else state.num_classes)
        self.distilling_flag=False
    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                ninp=self.state.ninp #Maybe 32
                out = self.encoder(x) * math.sqrt(ninp)
                #out=x
                #print(out.size())
                out.unsqueeze_(1)
                #print(out.size())
                out = F.relu(self.conv1(out), inplace=True)
        else:
                out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out
    
class TextConvNet2(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        if state.dropout:
            raise ValueError("TextConvNet2 doesn't support dropout")
        super(TextConvNet2, self).__init__()
        #if state.textdata:
        ninp=state.ninp #Maybe 32
        ntoken=state.ntoken
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.encoder.weight.requires_grad = False 
        self.conv1 = nn.Conv1d(state.maxlen, 250, 3)
        self.fc1 = nn.Linear(16, 250)
        self.fc2 = nn.Linear(250, 1 if state.num_classes <= 2 else state.num_classes)
        self.sigm=nn.Sigmoid()
        self.distilling_flag=False
    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                ninp=self.state.ninp #Maybe 32
                out = self.encoder(x) * math.sqrt(ninp)
                #out=x
                #print(out.size())
                #out.unsqueeze_(1)
                #print(out.size())
                out = F.relu(self.conv1(out), inplace=True)
        else:
                out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool1d(out, 3)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = self.sigm(self.fc2(out))
        return out    
class AlexCifarNet(utils.ReparamModule):
    supported_dims = {32}

    def __init__(self, state):
        super(AlexCifarNet, self).__init__()
        assert state.nc == 3
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


# ImageNet
class AlexNet(utils.ReparamModule):
    supported_dims = {224}

    class Idt(nn.Module):
        def forward(self, x):
            return x

    def __init__(self, state):
        super(AlexNet, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if state.dropout:
            filler = nn.Dropout
        else:
            filler = AlexNet.Idt
        self.classifier = nn.Sequential(
            filler(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            filler(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1 if state.num_classes <= 2 else state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
class PositionalEncoding(utils.ReparamModule):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, state, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(utils.ReparamModule):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    supported_dims = {28, 32}
    def __init__(self, state, ntoken=32, ninp=400, nhead=4, nhid=2, nlayers=2, dropout=0.5):
        ntoken=state.ntoken
        ninp=state.ninp
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(state, ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp) #Move to data preprocessing
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)  
