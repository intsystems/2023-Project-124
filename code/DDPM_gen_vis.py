import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
device = torch.device("cuda:0")


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


def generate_batch(ddpm, n_steps, n_samples, c, h, w):
    device = ddpm.device
    with torch.no_grad():
        x = torch.randn(n_samples, c, h, w).to(device)
        for idx, t in enumerate(tqdm(list(range(n_steps))[::-1], leave=False, desc="Steps", colour="#005500")):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor, n_steps)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
    return x


def generate_new_images(model=None, n_samples=16, device=None,
                         c=1, h=28, w=28, n_steps=None,
                        from_unet=False, nmax=3000):
    if from_unet:
        ddpm = MyDDPM(model, n_steps, device="cuda:0")
    else:
        ddpm = model

    ddpm.eval()

    torch.cuda.empty_cache()
    if n_steps is None:
        n_steps = ddpm.n_steps

    if n_samples <= nmax:
        x = generate_batch(ddpm, n_steps, n_samples, c, h, w)
    else:
        k = n_samples // nmax
        k += 1 if (n_samples % nmax != 0) else 0
        xs = []
        for i in tqdm(range(k), leave=False, desc="batches", colour="#000077"):
            torch.cuda.empty_cache()
            if i < k - 1 or (n_samples % nmax == 0):
                xs.append(generate_batch(ddpm, n_steps, nmax, c, h, w))
            else:
                xs.append(generate_batch(ddpm, n_steps, n_samples % nmax, c, h, w))
        x = torch.cat(xs, dim=0)

    try:
        del ddpm
    except: pass

    torch.cuda.empty_cache()

    return x


# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02,
                 device=None, image_chw=(1, 28, 28), emb_dim=300):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
        self.emb_matr = sinusoidal_embedding(n_steps, emb_dim).to(device)
        self.emb_dim = emb_dim

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t, n_st=None):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        if n_st is None:
            n_st = self.n_steps
            emb_matr = self.emb_matr
        else:
            emb_matr = sinusoidal_embedding(n_st, self.emb_dim).to(self.device)

        t = F.embedding(t, emb_matr)
        t.requires_grad_(False)

        return self.network(x, t)


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10000 ** (j * 2 / d) for j in range(d//2)])

    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk)
    embedding[:, 1::2] = torch.cos(t * wk)

    return embedding



