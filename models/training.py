# A training file for VAE model
# Dataset Loading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.datasets as datasets  
from torch.utils.data import DataLoader  
from encoder import VAE_Encoder
from decoder import VAE_Decoder


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim=200):
        super().__init__()
        # # encoder
        # self.img_2hid = nn.Linear(input_dim, h_dim)

        # # one for mu and one for stds, note how we only output
        # # diagonal values of covariance matrix. Here we assume
        # # the pixels are conditionally independent 
        # self.hid_2mu = nn.Linear(h_dim, z_dim)
        # self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # # decoder
        # self.z_2hid = nn.Linear(z_dim, h_dim)
        # self.hid_2img = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h, mu, sigma = VAE_Encoder(x)
        return mu, sigma

    def decode(self, z):
        x = VAE_Decoder(z)
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)

        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon

        x = self.decode(z_reparametrized)
        return x, mu, sigma







# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 160*80
Z_DIM = 200
H_DIM = 400
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4



# Dataset loading
dataset = '/Volumes/'
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)



# Define train function
def train(num_epochs, model, optimizer, loss_fn):
    # Start training
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM)
            x_reconst, mu, sigma = model(x)

            # loss
            reconst_loss = loss_fn(x_reconst, x)
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())



# Initialize model, optimizer, loss
model = VAE_Encoder(INPUT_DIM, Z_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

### change loss function to cross entropy
loss_fn = nn.BCELoss(reduction="sum")

# Run training
train(NUM_EPOCHS, model, optimizer, loss_fn)



### Inference
def inference(sample, num_examples=1):
    """
    Generates (num_examples)
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings.append((mu, sigma))

    mu, sigma = encodings[sample]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{sample}_ex{example}.png")
