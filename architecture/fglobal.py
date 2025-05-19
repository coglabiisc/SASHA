import torch
from torch import nn


class f_global(nn.Module):
    def __init__(self, in_features=384, nhead=4, num_layers=1, ff_dim=512, dropout=0.2):
        super(f_global, self).__init__()
        self.in_features = in_features
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.in_features,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        self.in_projection = nn.Linear(2*self.in_features, self.in_features)
        self.layer_norm = nn.LayerNorm(self.in_features)


    def forward(self, feature, state):
        """
        feature is a 1*d tensor of zoomed patch and state is a 1*N*d tensor
        based on the zoomed in information and the current state, we are updating the entire state
        """
        B,N,d = state.shape
        feature = feature.repeat(1, N, 1)
        state = torch.concat(tensors=(state, feature), dim=2)
        projected_state = nn.LeakyReLU()(self.layer_norm(self.in_projection(state)))
        op = self.encoder(projected_state)
        return op
    