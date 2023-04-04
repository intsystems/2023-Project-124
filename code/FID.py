import torch.nn as nn
import torch
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms, datasets
from scipy import linalg
import numpy as np
import glob
from torch.utils.data import Dataset
from PIL import Image
device = torch.device("cuda:0")


# Define the generator for the Inception v3 model
class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.inception = torchvision.models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
        self.inception.fc = nn.Identity()

    def forward(self, x):
        x = self.inception(x)
        return x


# Define the FID calculation function
def calculate_fid(real_features, fake_features):
    # Calculate the mean and covariance of the real and fake feature vectors
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # Calculate the squared sum of differences between means
    ssdiff = (mu1 - mu2).dot(mu1 - mu2)

    # Calculate the matrix square root of product of covariances
    eps = 1e-8
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # Check if any nan values are generated during matrix square root calculation
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


# Define the function to extract features from the Inception v3 model
def extract_features(loader, batch_size=50, num_classes=2048):
    # Set up the Inception v3 model
    model = InceptionV3().to(device)
    model.eval()

    # Extract the features for the images
    num_images = len(loader.dataset)
    features = np.zeros((num_images, num_classes))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, leave=False, desc="Feature extracting", colour="#005500")):
            batch = batch.cuda()
            batch_features = model(batch).squeeze()
            start = i * batch_size
            end = start + batch_size
            features[start:end] = batch_features.cpu().numpy()

    return features


class TestDataset(Dataset):
    def __init__(self, folder='data/test/', transform=None):
        self.transform = transform
        self.folder = folder
        self.paths = [f for f in glob.glob(folder + '*')]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img


def FID_score(path_real, path_fake, real_saved=False, num_features=2048):

    # Define the transforms for the images
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the dataloaders for the real and fake images
    if not real_saved:
        real_dataset = TestDataset(path_real, transform=transform)
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=50, shuffle=False)
        real_features = extract_features(real_loader, num_features)
    else:
        with open(path_real, 'rb') as f:
            real_features = np.load(f)

    fake_dataset = TestDataset(path_fake, transform=transform)
    fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=50, shuffle=False)

    # Extract the features for the real and fake images

    fake_features = extract_features(fake_loader, batch_size=50, num_classes=num_features)

    # Calculate the FID score
    fid = calculate_fid(real_features, fake_features)

    return fid


def extract_real_features(path_real):

    transform_ = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    real_dataset = TestDataset(path_real, transform=transform_)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=50, shuffle=False)

    return extract_features(real_loader)
