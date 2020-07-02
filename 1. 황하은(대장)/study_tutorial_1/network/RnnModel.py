import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(torch.nn.Module):
    def __init__(self, args):
        super(LSTMClassifier, self).__init__()
        self.args = args
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.layer_dim = self.args.layer_dim
        self.output_dim = self.args.output_dim
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim,
                           self.layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # hidden state는 가장 마지막 값 만 linear 층에 들어감
        return out

    def init_hidden(self, x):
        '''hidden state, cell state 초기값 만들기'''
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)  # hidden state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)  # cell state
        return [t.cuda() for t in (h0, c0)]