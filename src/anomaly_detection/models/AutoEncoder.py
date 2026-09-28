import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(dims, activation, batch_norm=False):
    layers = []
    if activation == "relu":
        act_layer = nn.ReLU
    elif activation == "gelu":
        act_layer = nn.GELU
    else:
        raise ValueError(f"Unknown activation '{activation}'. Use 'relu' or 'gelu'.")

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_layer())
    return nn.Sequential(*layers)


def _resolve_hidden_dims(seq_len, hidden_ratios):
    if hidden_ratios is None:
        return []
    return [int(round(float(r) * seq_len)) for r in hidden_ratios]


class Model(nn.Module):
    """
    Simple linear autoencoder for sequences.
    - Input:  [B, L, D]
    - Output: [B, L, D]
    """

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.individual = individual

        self.hidden_dims = _resolve_hidden_dims(
            self.seq_len,
            getattr(configs, "hidden_ratios", []),
        )

        self.activation = getattr(configs, "activation", "relu")
        self.batch_norm = getattr(configs, "batch_norm", False)

        # Encoder dims: L → ... → latent
        self.dims = [self.seq_len] + self.hidden_dims
        self.latent_len = self.dims[-1]

        if self.individual:
            self.Enc = nn.ModuleList()
            self.Dec = nn.ModuleList()

            for _ in range(self.channels):
                self.Enc.append(
                    _build_mlp(self.dims, self.activation, self.batch_norm)
                )
                self.Dec.append(
                    _build_mlp(list(reversed(self.dims)), self.activation, self.batch_norm)
                )
        else:
            self.Enc = _build_mlp(self.dims, self.activation, self.batch_norm)
            self.Dec = _build_mlp(list(reversed(self.dims)), self.activation, self.batch_norm)

    def encode(self, x):
        """
        x: [B, L, D]
        returns: [B, latent_len, D]
        """
        B, L, D = x.shape
        x = x.permute(0, 2, 1)  # [B, D, L]

        if self.individual:
            latent = []
            for i in range(self.channels):
                out = self.Enc[i](x[:, i, :])  # [B, latent_len]
                latent.append(out.unsqueeze(1))
            latent = torch.cat(latent, dim=1)  # [B, D, latent_len]
        else:
            x = x.reshape(B * D, L)
            latent = self.Enc(x)               # [B*D, latent_len]
            latent = latent.view(B, D, -1)    # [B, D, latent_len]

        return latent.permute(0, 2, 1)  # [B, latent_len, D]

    def decode(self, z):
        """
        z: [B, latent_len, D]
        returns: [B, L, D]
        """
        B, latent_len, D = z.shape
        z = z.permute(0, 2, 1)  # [B, D, latent_len]

        if self.individual:
            x_rec = []
            for i in range(self.channels):
                out = self.Dec[i](z[:, i, :])  # [B, L]
                x_rec.append(out.unsqueeze(1))
            x_rec = torch.cat(x_rec, dim=1)  # [B, D, L]
        else:
            z = z.reshape(B * D, latent_len)
            x_rec = self.Dec(z)              # [B*D, L]
            x_rec = x_rec.view(B, D, -1)     # [B, D, L]

        return x_rec.permute(0, 2, 1)  # [B, L, D]

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)