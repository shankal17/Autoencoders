# Autoencoders

Exercises with autoencoders. Both fully-connected and fully-convolutional autoencoders were implemented.

## Results
### Denoising Autoencoder Result

The top row is comprised of the input images with gaussian noise added. The bottom row are the corresponding outputs of the autoencoder. Note that most of the noise is removed.

![denoising](/results/denoising_autoencoder_result.PNG)

### Normal Autoencoder Result

The top row is comprised of the input images that are passed into the autoencoder. The bottom row are the corresponding outputs of the autoencoder. Notice that they look very similar to the inputs. This is what we want.

![normal](/results/convolutional_autoencoder_result.PNG)

## Usage
Once you have the code, set up a virtual environment if you would like and install the necessary libraries by running the command below.
```bat
pip install -r /path/to/requirements.txt
```
If you want to change any of the architecures, they are located in [models.py](https://github.com/shankal17/Autoencoders/blob/main/models/models.py).

Then, following the process in [denoising_autoencoder_example.ipynb](https://github.com/shankal17/Autoencoders/blob/main/notebooks/denoising_autoencoder_example.ipynb) train your own autoencoders.
