import os
import dnnlib
import legacy
import torch

from facenet_encoder import dataset_creator


def import_generator(generator_pkl, device=torch.device('cpu')):
    """
    Create a dataset of generated images and their corresponding latent vectors using a trained generator.
    Args:
        generator_pkl:  Path to the pkl file from which to load the generator.
        device:         Torch device (GPU or CPU) to load the generator to.

    Returns: Instance of teh generator loaded from the pkl file.
    """
    g_kwargs = dnnlib.EasyDict()
    g_kwargs.size = None
    g_kwargs.scale_type = 'pad'
    with dnnlib.util.open_url(generator_pkl) as f:
        generator = legacy.load_network_pkl(f, custom=True, **g_kwargs)['G_ema'].to(device)  # type: ignore
    return generator


def create_cifar10_dataset():
    """
    Generate the dataset of images and their latent vectors using a cifar10 trained generator
    """
    # set up the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device {}'.format(device))

    # import the generator from a pkl file
    generator_pkl = os.path.join('..', 'pretrained', 'cifar10.pkl')
    print('Loading generator model from {}...'.format(generator_pkl))
    generator = import_generator(generator_pkl, device)

    # create the dataset using the cifar10 generator
    output_path = os.path.join('..', 'out', 'cifar10_ds')
    print('Creating dataset at {}...'.format(output_path))
    dataset_creator.create_encoder_dataset(generator, 10, output_path, device)

    print('Dataset created.')


if __name__ == '__main__':
    create_cifar10_dataset()
