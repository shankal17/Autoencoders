import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm

class GaussianNoise(object):
    """Class that adds gaussian noise to a tensor
    ...
    
    Attributes
    ----------
    mean : float, optional
        Mean of the gaussian noise
    std : float, optional
        Standard deviation of the gaussian noise
    
    Methods
    -------
    __call__(idx)
        Returns tensor with gaussian noise added to input
    __repr__()
        Returns class attributes
    """

    def __init__(self, mean=0, std=1.0):
        self.std = std
        self.mean = mean
    
    def __call__(self, x):
        x_noisy = x + torch.randn(x.size()) * self.std + self.mean
        x_noisy = np.clip(x_noisy, 0.0, 1.0)
        return x_noisy
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def train_denoising_autoencoder(autoencoder, train_loader, epochs=20, lr=0.001, learned_transform=None):
    """Trains a denoising autoencoder

    Parameters
    ----------
    autoencoder : torch.nn.Module
        Autoencoder to be trained
    train_loader : torch.utils.data.DataLoader
        DataLoader with training data
    epochs : int, optional
        Number of epoch to train the model
    lr : float, optional
        Initial learning rate of Adam optimizer and
    learned_transform : function, optional
        Transform to run on the input data

    Returns
    -------
    torch.nn.Module
        Trained autoencoder
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    print('Training on ' + str(list(autoencoder.parameters())[0].device))
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        running_loss = 0.0

        for data in tqdm(train_loader):
            # Load images
            images, _ = data
            if learned_transform is not None:
                noisy_images = learned_transform(images)
                noisy_images = np.clip(noisy_images, 0.0, 1.0)
            images = images.to(device)
            noisy_images = noisy_images.to(device)

            # Zero optimizer gradients
            optimizer.zero_grad()

            # Pass noisy images through autoencoder
            denoised_images = autoencoder(noisy_images)

            # Compute loss
            loss = criterion(denoised_images, images)

            # Backpropagate
            loss.backward()

            # Update autoencoder
            optimizer.step()

            # Increment running loss
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(train_loader)
        print('epoch: {} Loss: {:.6f}'.format(epoch, epoch_loss))

    return autoencoder

def train_throughput_autoencoder(autoencoder, train_loader, epochs=20, lr=0.001):
    """Trains an autoencoder

    Parameters
    ----------
    autoencoder : torch.nn.Module
        Autoencoder to be trained
    train_loader : torch.utils.data.DataLoader
        DataLoader with training data
    epochs : int, optional
        Number of epoch to train the model
    lr : float, optional
        Initial learning rate of Adam optimizer and

    Returns
    -------
    torch.nn.Module
        Trained autoencoder
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    print('Training on ' + str(list(autoencoder.parameters())[0].device))

    # Define criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        running_loss = 0.0

        for data in tqdm(train_loader):
            # Load images and push them to device
            images, _ = data
            images = images.to(device)

            # Zero optimizer gradients
            optimizer.zero_grad()

            # Pass noisy images through autoencoder
            reconstructed_images = autoencoder(images)

            # Compute loss
            loss = criterion(reconstructed_images, images)

            # Backpropagate
            loss.backward()

            # Update autoencoder
            optimizer.step()

            # Increment running loss
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(train_loader)
        print('epoch: {} Loss: {:.6f}'.format(epoch, epoch_loss))

    return autoencoder

def visualize_inputs_outputs(autoencoder, data_loader, num_columns=4, learned_transform=None):
    """Show inputs and outputs of autoencoder

    Parameters
    ----------
    autoencoder : torch.nn.Module
        Autoencoder which data is being run
    data_loader : torch.utils.data.DataLoader
        DataLoader with input data
    num_columns : int, optional
        Number of columns to display
    learned_transform : function, optional
        Transform that the autoencoder was trained to clean
    """

    # Always run on cpu for this cause it's easy and doesn't take long
    autoencoder.to("cpu")
    data_iterator = iter(data_loader)
    input_images, _ = data_iterator.next()

    # If denoising autoencoder, generate some data with same transfrom
    if learned_transform is not None:
        input_images = learned_transform(input_images)
        input_images = np.clip(input_images, 0.0, 1.0)
    
    # Run inference
    outputs = autoencoder(input_images)

    # Prepare for plotting
    input_images = input_images.numpy()
    outputs = outputs.view(data_loader.batch_size, 1, 28, 28) # Assumed mnist dataset
    output = outputs.detach().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=num_columns, sharex=True, sharey=True, figsize=(num_columns*3, 6))

    # Plot the inputs and outputs
    for input_images, row in zip([input_images, output], axes):
        for img, ax in zip(input_images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)



if __name__ == '__main__':
    gn = GaussianNoise()
    print(gn)