import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import imageio
import einops
device = torch.device("cuda:0")


def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


    def generate_new_images(model=None, n_samples=16, device=None, frames_per_gif=100,
                            gif_name="sampling.gif", c=1, h=28, w=28, n_steps=None, gif=False,
                            from_unet=False):
        """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
        if from_unet:
            ddpm = MyDDPM(model, n_steps, device=device)
        else:
            ddpm = model
        torch.cuda.empty_cache()

        frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
        frames = []

        with torch.no_grad():
            if device is None:
                device = ddpm.device

            # Starting from random noise
            x = torch.randn(n_samples, c, h, w).to(device)

            for idx, t in enumerate(tqdm(list(range(ddpm.n_steps))[::-1], leave=False, desc="Steps", colour="#005500")):
                # Estimating noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
                eta_theta = ddpm.backward(x, time_tensor, n_steps)

                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(n_samples, c, h, w).to(device)

                    # Option 1: sigma_t squared = beta_t
                    beta_t = ddpm.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z

                # Adding frames to the GIF
                if gif:
                    if idx in frame_idxs or t == 0:
                        # Putting digits in range [0, 255]
                        normalized = x.clone()
                        for i in range(len(normalized)):
                            normalized[i] -= torch.min(normalized[i])
                            normalized[i] *= 255 / torch.max(normalized[i])

                        # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                        frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c",
                                                 b1=int(n_samples ** 0.5))
                        frame = frame.cpu().numpy().astype(np.uint8)

                        # Rendering frame
                        frames.append(frame)

        # Storing the gif
        if gif:
            with imageio.get_writer(gif_name, mode="I") as writer:
                for idx, frame in enumerate(frames):
                    writer.append_data(frame)
                    if idx == len(frames) - 1:
                        for _ in range(frames_per_gif // 3):
                            writer.append_data(frames[-1])

        try:
            del ddpm
        except:
            pass
        torch.cuda.empty_cache()

        return x

    # DDPM class
    class MyDDPM(nn.Module):
        def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
            super(MyDDPM, self).__init__()
            self.n_steps = n_steps
            self.device = device
            self.image_chw = image_chw
            self.network = network.to(device)
            self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
                device)  # Number of steps is typically in the order of thousands
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(
                device)
            self.emb_matr = sinusoidal_embedding(n_steps, 100).to(device)

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
                emb_matr = sinusoidal_embedding(n_st, 100).to(self.device)

            t = F.embedding(t, emb_matr)
            t.requires_grad_(False)

            return self.network(x, t)


    def sinusoidal_embedding(n, d):
        # Returns the standard positional embedding
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

        return embedding

