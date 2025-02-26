import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from torch.utils.data import DataLoader
from collections import OrderedDict

# Set random seed for reproducibility
seed = 999
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Root directory for dataset
data_root = "D:\celeb_dataset\images"
os.makedirs(data_root, exist_ok=True)

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images
image_size = 64

# Number of channels in the training images
nc = 3

# Size of z latent vector (generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5  # ---< Change this

# Learning rate for Adam optimizers (still used alongside GA)
lr = 0.0002

# Beta1 hyperparameter for Adam optimizer
beta1 = 0.5

# Genetic Algorithm Parameters
population_size = 10  # Number of G-D network pairs in population
elite_size = 2  # Number of top networks to keep unchanged
mutation_rate = 0.05  # Probability of mutating a weight
crossover_rate = 0.7  # Probability of performing crossover
tournament_size = 3  # Number of individuals in tournament selection
ga_generations = 5  # Number of GA generations per epoch

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Download and preprocess CelebA dataset


def get_celeba_dataset():
    # Create transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create dataset
    dataset = dset.ImageFolder(
        root=os.path.dirname(data_root), transform=transform)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)

    return dataloader, dataset

# Generator Network


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Network


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Initialize model weights


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Individual in the genetic algorithm (contains both G and D networks)


class GANIndividual:
    def __init__(self, g_net=None, d_net=None):
        # Initialize or copy networks
        if g_net is None:
            self.g_net = Generator().to(device)
            self.g_net.apply(weights_init)
        else:
            self.g_net = copy.deepcopy(g_net)

        if d_net is None:
            self.d_net = Discriminator().to(device)
            self.d_net.apply(weights_init)
        else:
            self.d_net = copy.deepcopy(d_net)

        self.fitness = 0.0

    def get_g_state_dict(self):
        return self.g_net.state_dict()

    def get_d_state_dict(self):
        return self.d_net.state_dict()

    def set_g_state_dict(self, state_dict):
        self.g_net.load_state_dict(state_dict)

    def set_d_state_dict(self, state_dict):
        self.d_net.load_state_dict(state_dict)

# Initialize population of G-D network pairs


def initialize_population(size):
    return [GANIndividual() for _ in range(size)]

# Tournament selection


def tournament_selection(population):
    selected_indices = random.sample(range(len(population)), tournament_size)
    tournament = [population[i] for i in selected_indices]
    tournament.sort(key=lambda ind: ind.fitness, reverse=True)
    return tournament[0]

# Crossover two state dictionaries


def crossover_state_dicts(sd1, sd2):
    if random.random() > crossover_rate:
        return copy.deepcopy(sd1)

    new_sd = OrderedDict()

    # For each layer's parameters
    for key in sd1:
        # Decide crossover method per layer randomly
        method = random.randint(0, 2)

        if method == 0:
            # Take all from parent 1
            new_sd[key] = sd1[key].clone()
        elif method == 1:
            # Take all from parent 2
            new_sd[key] = sd2[key].clone()
        else:
            # Perform element-wise crossover at random split point
            param1 = sd1[key]
            param2 = sd2[key]
            split_point = random.randint(0, param1.numel() - 1)

            # Convert to 1D for easier manipulation
            p1_flat = param1.flatten()
            p2_flat = param2.flatten()

            # Create new parameter tensor
            new_param = p1_flat.clone()
            new_param[split_point:] = p2_flat[split_point:]

            # Reshape back to original shape
            new_sd[key] = new_param.reshape(param1.shape)

    return new_sd

# Mutate a state dictionary


def mutate_state_dict(sd):
    new_sd = OrderedDict()
    for key, param in sd.items():
        # Only mutate if the parameter is floating point.
        if torch.is_floating_point(param):
            mask = torch.rand_like(param) < mutation_rate
            if mask.sum() > 0:  # If any element should be mutated
                mutation_strength = 0.1  # Scale of mutations
                noise = torch.randn_like(param) * mutation_strength
                param[mask] += noise[mask]
        # For non-floating point parameters, just copy them
        new_sd[key] = param
    return new_sd

# Evaluate fitness of an individual on a batch of real images


def evaluate_fitness(individual, real_images, criterion):
    g_net = individual.g_net
    d_net = individual.d_net

    batch_size = real_images.size(0)

    with torch.no_grad():
        # Create labels
        real_label = torch.full(
            (batch_size,), 1.0, dtype=torch.float, device=device)
        fake_label = torch.full(
            (batch_size,), 0.0, dtype=torch.float, device=device)

        # Forward pass with real images
        output_real = d_net(real_images).view(-1)
        loss_d_real = criterion(output_real, real_label)
        d_x = output_real.mean().item()

        # Generate fake images
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = g_net(noise)

        # Forward pass with fake images
        output_fake = d_net(fake_images).view(-1)
        loss_d_fake = criterion(output_fake, fake_label)
        d_g_z = output_fake.mean().item()

        # Discriminator performance
        loss_d = loss_d_real + loss_d_fake
        # Higher when D correctly classifies both real and fake
        d_performance = d_x * (1 - d_g_z)

        # Generator performance - fool the discriminator
        loss_g = criterion(output_fake, real_label)
        g_performance = output_fake.mean().item()  # Higher when D is fooled

        # Image quality metrics (simple version - can be enhanced)
        # Check for mode collapse (variety in fake images)
        fake_std = fake_images.std().item()

        # Combined fitness score (weighted sum)
        # Balance between:
        # 1. Discriminator correctly classifying real/fake
        # 2. Generator fooling discriminator
        # 3. Quality of generated images (avoid mode collapse)

        fitness = (0.3 * d_performance +  # Good discrimination
                   0.4 * g_performance +   # Generator fooling discriminator
                   0.3 * fake_std)         # Image diversity (avoid mode collapse)

        return fitness

# Create next generation through GA operations


def create_next_generation(population, elites_kept):
    # Sort by fitness
    population.sort(key=lambda ind: ind.fitness, reverse=True)

    # Keep elites
    next_gen = population[:elites_kept]

    # Create the rest through selection, crossover, and mutation
    while len(next_gen) < len(population):
        # Selection
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        # Create a new individual
        child = GANIndividual()

        # Crossover generator networks
        g_state_dict = crossover_state_dicts(
            parent1.get_g_state_dict(),
            parent2.get_g_state_dict()
        )

        # Crossover discriminator networks
        d_state_dict = crossover_state_dicts(
            parent1.get_d_state_dict(),
            parent2.get_d_state_dict()
        )

        # Mutation
        g_state_dict = mutate_state_dict(g_state_dict)
        d_state_dict = mutate_state_dict(d_state_dict)

        # Update the child's networks
        child.set_g_state_dict(g_state_dict)
        child.set_d_state_dict(d_state_dict)

        next_gen.append(child)

    return next_gen

# Main training function using genetic algorithm


def train_dcgan_with_ga():
    print("Starting DCGAN training with Genetic Algorithm for network weights...")

    # Get dataset
    dataloader, _ = get_celeba_dataset()

    # Create initial population
    population = initialize_population(population_size)

    # Loss function
    criterion = nn.BCELoss()

    # For visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Tracking variables
    img_list = []
    G_losses = []
    D_losses = []

    # Create optimizers for the best individual (for Adam updates)
    best_individual = population[0]
    optimizerD = optim.Adam(
        best_individual.d_net.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(
        best_individual.g_net.parameters(), lr=lr, betas=(beta1, 0.999))

    # Main training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # Step 1: Evaluate fitness of all individuals
            for ind in population:
                ind.fitness = evaluate_fitness(ind, real_cpu, criterion)

            # Step 2: Sort population by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Update best individual (will be used for visualization and Adam update)
            best_individual = population[0]

            # Step 3: Apply genetic algorithm to evolve population
            # Run GA every few batches to save computation
            if i % 30 == 0:
                print(
                    f"  Batch {i}/{len(dataloader)}: Running genetic algorithm...")

                for gen in range(ga_generations):
                    # Create next generation
                    population = create_next_generation(population, elite_size)

                    # Evaluate fitness of new population on the same batch
                    for ind in population:
                        ind.fitness = evaluate_fitness(
                            ind, real_cpu, criterion)

                # Sort population again after evolution
                population.sort(key=lambda ind: ind.fitness, reverse=True)
                best_individual = population[0]

                # Reset optimizers for the best individual
                optimizerD = optim.Adam(
                    best_individual.d_net.parameters(), lr=lr, betas=(beta1, 0.999))
                optimizerG = optim.Adam(
                    best_individual.g_net.parameters(), lr=lr, betas=(beta1, 0.999))

            # Step 4: Also apply Adam optimizer on the best individual
            # This combines genetic algorithm with gradient-based optimization

            # Update discriminator with Adam
            best_individual.d_net.zero_grad()

            # Train with real batch
            label = torch.full(
                (b_size,), 1.0, dtype=torch.float, device=device)
            output = best_individual.d_net(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = best_individual.g_net(noise)
            label.fill_(0.0)
            output = best_individual.d_net(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Update generator with Adam
            best_individual.g_net.zero_grad()
            label.fill_(1.0)
            output = best_individual.d_net(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
                print(f"Best fitness: {best_individual.fitness:.4f}")

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = best_individual.g_net(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(
                    fake, padding=2, normalize=True))

        # Save model checkpoints after each epoch
        torch.save(best_individual.g_net.state_dict(),
                   f'generator_ga_weights_epoch_{epoch}.pth')
        torch.save(best_individual.d_net.state_dict(),
                   f'discriminator_ga_weights_epoch_{epoch}.pth')

        # Visualize generator progress
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.title(f"Generated Images - Epoch {epoch}")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig(f"ga_generated_epoch_{epoch}.png")
        plt.close()

    # Visualization after training
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training (with GA)")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("dcgan_ga_weights_loss_plot.png")

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
               :64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images (with GA)")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig("dcgan_ga_weights_final_comparison.png")

    return best_individual.g_net, best_individual.d_net, G_losses, D_losses, img_list


if __name__ == "__main__":
    # Mohamed Fathallah Currently working version but not good results so far.
    train_dcgan_with_ga()
