import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1,
                 padding=1, activation=None, normalize=True):
        super(DoubleConv, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class UNetSmall(nn.Module):
    def __init__(self, time_emb_dim=300, in_channels=1, out_channels=1):
        super(UNetSmall, self).__init__()

        # First half
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.b1 = nn.Sequential(
            DoubleConv((in_channels, 28, 28), in_channels, 15),
            DoubleConv((15, 28, 28), 15, 15),
            DoubleConv((15, 28, 28), 15, 15)
        )
        self.down1 = nn.MaxPool2d(2, 2)

        self.te2 = self._make_te(time_emb_dim, 15)
        self.b2 = nn.Sequential(
            DoubleConv((15, 14, 14), 15, 30),
            DoubleConv((30, 14, 14), 30, 30),
            DoubleConv((30, 14, 14), 30, 30)
        )
        self.down2 = nn.MaxPool2d(2, 2)

        self.te3 = self._make_te(time_emb_dim, 30)
        self.b3 = nn.Sequential(
            DoubleConv((30, 7, 7), 30, 60),
            DoubleConv((60, 7, 7), 60, 60),
            DoubleConv((60, 7, 7), 60, 60)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(60, 60, 2, 1),
            nn.SiLU(),
            nn.Conv2d(60, 60, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 60)
        self.b_mid = nn.Sequential(
            DoubleConv((60, 3, 3), 60, 120),
            DoubleConv((120, 3, 3), 120, 120),
            DoubleConv((120, 3, 3), 120, 60)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(60, 60, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(60, 60, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 120)
        self.b4 = nn.Sequential(
            DoubleConv((120, 7, 7), 120, 60),
            DoubleConv((60, 7, 7), 60, 30),
            DoubleConv((30, 7, 7), 30, 30)
        )

        self.up2 = nn.ConvTranspose2d(30, 30, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 60)
        self.b5 = nn.Sequential(
            DoubleConv((60, 14, 14), 60, 30),
            DoubleConv((30, 14, 14), 30, 15),
            DoubleConv((15, 14, 14), 15, 15)
        )

        self.up3 = nn.ConvTranspose2d(15, 15, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 30)
        self.b_out = nn.Sequential(
            DoubleConv((30, 28, 28), 30, 15),
            DoubleConv((15, 28, 28), 15, 15),
            DoubleConv((15, 28, 28), 15, 15, normalize=False)
        )

        self.conv_out = nn.Conv2d(15, out_channels, 3, 1, 1)

    def forward(self, x, t):

        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10000 ** (j * 2 / d) for j in range(d//2)])

    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk)
    embedding[:, 1::2] = torch.cos(t * wk)

    return embedding
