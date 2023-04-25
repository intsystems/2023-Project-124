import torch.nn as nn
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
device = torch.device("cuda:0")


class DDGAN(nn.Module):
    def __init__(self, generator, discriminator, n_steps=15,
                 min_beta=2e-1, max_beta=9e-1, device=None,
                 image_chw=(1, 28, 28), emb_dim=100):
        super(DDGAN, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        first = torch.tensor(1e-8).to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.betas = torch.cat((first[None], self.betas)).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
        self.sigmas_cum = (1 - self.alpha_bars) ** 0.5
        self.emb_matr = sinusoidal_embedding(n_steps, emb_dim).to(device)
        self.emb_dim = emb_dim

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    # def backward(self, x, t, n_st=None):
    #     if n_st is None:
    #         n_st = self.n_steps
    #         emb_matr = self.emb_matr
    #     else:
    #         emb_matr = sinusoidal_embedding(n_st, self.emb_dim).to(self.device)
    #
    #     t = F.embedding(t, emb_matr)
    #     t.requires_grad_(False)
    #
    #     latent_z = torch.randn_like(x).to(device)
    #
    #     return self.generator(x, t, latent_z)


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 100 ** (j * 2 / d) for j in range(d//2)])

    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk)
    embedding[:, 1::2] = torch.cos(t * wk)

    return embedding


def extract(inp, t, shape):
    out = torch.gather(inp, 0, t)

    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def q_sample(ddgan, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(ddgan.alpha_bars ** 0.5, t, x_start.shape) * x_start + \
          extract(ddgan.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(ddgan, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(ddgan, x_start, t)

    x_t_plus_one = x_t * extract(ddgan.alphas ** 0.5, t + 1, x_start.shape) + \
                   extract(ddgan.betas ** 0.5, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


class Posterior_Coefficients():
    def __init__(self, ddgan, device):
        self.betas = ddgan.betas

        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos



def generate_batch(ddgan, n_steps, n_samples, c, h, w):
    with torch.no_grad():
        device = ddgan.device
        pos_coeff = Posterior_Coefficients(ddgan, device)
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(tqdm(list(range(n_steps))[::-1], leave=False, desc="Steps", colour="#006600")):
            time_tensor = (torch.ones(n_samples, ) * t).to(device).long()
            latent_z = torch.randn(n_samples, ddgan.generator.zsize).to(device)

            x_0_predict = ddgan.generator(x, time_tensor, latent_z)
            x = sample_posterior(pos_coeff, x_0_predict, x, time_tensor).detach()
    return x


def generate_new_images(model=None, n_samples=16, device=None,
                        c=1, h=28, w=28, n_steps=None, nmax=2000):
    ddgan = model
    torch.cuda.empty_cache()
    ddgan.eval()

    if device == None:
        device = ddgan.device

    if n_steps is None:
        n_steps = ddgan.n_steps

    if n_samples <= nmax:
        x = generate_batch(ddgan, n_steps, n_samples, c, h, w)
    else:
        k = n_samples // nmax
        k += 1 if (n_samples % nmax != 0) else 0
        xs = []
        for i in tqdm(range(k), leave=False, desc="batches", colour="#000077"):
            torch.cuda.empty_cache()
            if i < k - 1 or (n_samples % nmax == 0):
                xs.append(generate_batch(ddgan, n_steps, nmax, c, h, w))
            else:
                xs.append(generate_batch(ddgan, n_steps, n_samples % nmax, c, h, w))
        x = torch.cat(xs, dim=0)

    try:
        del ddgan
    except:
        pass

    torch.cuda.empty_cache()

    return x


def show_images(images, save_path=None):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.rcParams['axes.grid'] = False

    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1, xticks=[], yticks=[])

            if idx < len(images):

                img = images[idx][0]
                img -= img.min()
                img /= img.max()

                plt.imshow(img, cmap="gray")
                idx += 1

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Showing the figure
    plt.show()
