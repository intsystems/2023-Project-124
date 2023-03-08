import torch.nn as nn
import torch


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


def _make_te(self, dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.SiLU(),
        nn.Linear(dim_out, dim_out)
    )


class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, 
                 padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
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


class UNetLarge(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, in_channels=3, out_channels=3):
        super(UNetLarge, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.b1 = nn.Sequential(
            MyBlock((in_channels, 32, 32), in_channels, 64),
            MyBlock((64, 32, 32), 64, 64),
            MyBlock((64, 32, 32), 64, 64)
        )

        self.down1 = nn.MaxPool2d(2, 2)

        self.te2 = self._make_te(time_emb_dim, 64)
        self.b2 = nn.Sequential(
            MyBlock((64, 16, 16), 64, 128),
            MyBlock((128, 16, 16), 128, 128),
            MyBlock((128, 16, 16), 128, 128)
        )

        self.down2 = nn.MaxPool2d(2, 2)

        self.te3 = self._make_te(time_emb_dim, 128)
        self.b3 = nn.Sequential(
            MyBlock((128, 8, 8), 128, 256),
            MyBlock((256, 8, 8), 256, 256),
            MyBlock((256, 8, 8), 256, 256)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 2, 1),
            nn.SiLU(),
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 256)
        self.b_mid = nn.Sequential(
            MyBlock((256, 3, 3), 256, 512),
            MyBlock((512, 3, 3), 512, 1024),
            MyBlock((1024, 3, 3), 1024, 512),
            MyBlock((512, 3, 3), 512, 256)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 256, 3, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 512)
        self.b4 = nn.Sequential(
            MyBlock((512, 8, 8), 512, 256),
            MyBlock((256, 8, 8), 256, 256),
            MyBlock((256, 8, 8), 256, 128)
        )

        self.up2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 256)
        self.b5 = nn.Sequential(
            MyBlock((256, 16, 16), 256, 128),
            MyBlock((128, 16, 16), 128, 128),
            MyBlock((128, 16, 16), 128, 64)
        )

        self.up3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 128)
        self.b_out = nn.Sequential(
            MyBlock((128, 32, 32), 128, 64),
            MyBlock((64, 32, 32), 64, 64),
            MyBlock((64, 32, 32), 64, 64, normalize=False)
        )

        self.conv_out = nn.Conv2d(64, out_channels, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_embed(t)
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


