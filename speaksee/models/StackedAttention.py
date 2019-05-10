import torch
from torch import nn
from torch.nn import functional as F

device = torch.device('cuda')


class LstmEncoder(nn.Module):
    """
    LSTM encoder
    Embedding(question) -> RNN()

    """

    def __init__(self, vocab_size, rnn_dim, wordvec_dim=500,
                 rnn_num_layers=2, rnn_dropout=0):
        super(LstmEncoder, self).__init__()

        self.NULL = 1
        self.START = 0
        self.END = 2

        self.rnn_size = rnn_dim
        self.embed = nn.Embedding(vocab_size, wordvec_dim)
        self.rnn = nn.LSTM(wordvec_dim, rnn_dim, rnn_num_layers,
                           dropout=rnn_dropout, batch_first=True, bidirectional=True)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in RNN and Embedding layers

        """

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            if 'bias' in name:
                param.data.fill_(0)
        nn.init.kaiming_uniform_(self.embed.weight)

    def forward(self, x):
        # type: (torch.Tensor) -> (torch.Tensor, torch.Tensor)
        """
        Forward Pass

        x: question batch_size x question_len

        return: hs = question (batch_size X question_len X rnn_dim)
                idx = question-length, without padding (batch_size x 1)

        """

        N, T = x.shape
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        x_cpu = x.data.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data).long()
        idx.requires_grad = False

        hs, _ = self.rnn(self.embed(x))

        hs = hs.view(N, T, self.rnn_size, 2)

        # Split in forward and backward sequence
        output_forward, output_backward = torch.chunk(hs, 2, 3)
        output_forward = output_forward.squeeze(3)  # (batch_size, T, hidden_size)
        output_backward = output_backward.squeeze(3)  # (batch_size, T, hidden_size)

        # Find last elements of the forward sequence
        q_len = idx.view(N, 1, 1).expand(N, 1, self.rnn_size).to(device)

        # Trunk the forward sequence at t = question_len
        output_forward = output_forward.gather(1, q_len).view(N, self.rnn_size)
        # Get last state of the backward sequence t = 0
        output_backward = output_backward[:, 0, :]
        # Re-concat output
        output = torch.cat((output_forward, output_backward), -1)

        return output


class StackedAttention(nn.Module):
    """
    Attention module, paper sez 3.3

    """

    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention, self).__init__()
        self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def getMap(self):
        """
        Get saved attention map

        return: Attention map

        """

        return torch.squeeze(self.attention_maps[0], 1)

    def forward(self, v, u):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward Pass

        v: visual features (N x D x H x W)
        u: attention key (N x D)
        return: attention map (N x D)

        """

        N, K = v.size(0), self.hidden_dim
        D, H, W = v.size(1), v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
        h = F.tanh(v_proj + u_proj_expand)
        p = F.softmax(self.Wp(h).view(N, H * W), -1).view(N, 1, H, W)
        self.attention_maps = p.data.clone()

        v_tilde = (p.expand_as(v) * v).sum(3).sum(2).view(N, D)
        return v_tilde


class SAN(nn.Module):
    """
    Stacked attention model
    paper https://arxiv.org/pdf/1511.02274.pdf

    """

    def __init__(self, vocab_size, n_answers, num_attention=2, rnn_size=256, att_size=512):
        super(SAN, self).__init__()

        print("----------- Build Neural Network -----------")

        # Params
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.n_answers = n_answers
        self.num_attention = num_attention

        # Layers
        self.rnn = LstmEncoder(vocab_size, self.rnn_size)
        self.stacked_attns = []
        for i in range(0, num_attention):
            sa = StackedAttention(self.att_size, self.att_size).to(device)
            self.stacked_attns.append(sa)

        self.classifier = nn.Sequential(nn.Linear(self.att_size, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, self.n_answers)  # note no softmax here
                                        )

    def forward(self, feats, question):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward pass

        feats: Image features (N x D x H x W)
        question: Question (N x L)
        return: Probability distribution over a set of answers

        """

        batch_size = feats.shape[0]

        # Question embedding
        q = self.rnn(question)

        # Attention steps
        u1 = q

        for sa in self.stacked_attns:
            u2 = sa(feats, u1)
            u2 = u2 + u1
            u1 = u2

        # Classifier
        out = self.classifier(u2)
        return out

    def get_data(self):
        """
        Get attention map, useful for visualization

        return: Tensor

        """
        return self.attention.getMap()
