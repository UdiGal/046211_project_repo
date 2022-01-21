import os
import pickle
import torch
from facenet_encoder.utils import save_img, rescale_img, one_hot_vector


def create_encoder_dataset(generator, nimages, output_path='./dataset/', device=torch.device('cpu')):
    """
    Create a dataset of generated images and their corresponding latent vectors using a trained generator.
    Args:
        generator:      Conditioned Generator of images. Receives a latent vector and a class, and generates an image.
        nimages:        Amount of images to generated for each class in the dataset.
        output_path:    Path to the directory that will hold the created dataset.
        device:         Device which the generator is loaded to.

    Returns: None
    """
    try:
        # get the size of the latent space that the generator accepts
        z_size = generator.z_dim
        # get the number of classes the generator accepts
        n_classes = generator.c_dim
        # get the shape of the images the generator outputs
        img_shape = generator.output_shape
    except AttributeError as err:
        print('Generator does not hold the appropriate fields: z_dim, c_dim, output_shape.')
        print(err)
        return

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'readme.txt'), 'w') as f:
        f.writelines(['Images - Latent Vectors Dataset:\n',
                      'Image dim:        {}\n'.format(img_shape),
                      'Latent dim:       {}\n'.format([1, z_size]),
                      'Number classes:   {}\n'.format(n_classes),
                      'Images per class: {}\n'.format(nimages),
                      'Total images:     {}\n'.format(nimages * n_classes),
                      'Dataset images format:    <class-index>_<image-index>.jpg\n',
                      'Latent dictionary format: key:   "<class-index>_<image-index>"\n',
                      '                          value: vector of latent variables {}\n'.format([1, z_size])])

    # create a dataset of images and latent vectors for each class
    os.makedirs(output_path, exist_ok=True)
    latent_dict = {}
    for class_idx in range(n_classes):
        # create a one-hot-vector of the current class
        class_vector = one_hot_vector(n_classes, class_idx, device)
        # create latent vectors and images for the class
        for img_idx in range(nimages):
            # random latent vector
            random_z = torch.randn([1, z_size]).to(device)
            # generated image from the latent vector and class
            gen_img = generator(random_z, class_vector)
            gen_img = rescale_img(gen_img).cpu().numpy()

            # save the latent vector and image to the dataset
            img_name = str(class_idx) + "_" + str(img_idx)
            latent_dict[img_name] = random_z
            save_img(gen_img, os.path.join(output_path, str(img_name) + '.jpg'))
        print('Created dataset for class {}'.format(class_idx))

    # save the latent dictionary to a pkl file
    latent_dict_name = 'latent_dict.pkl'
    with open(os.path.join(output_path, latent_dict_name), 'wb') as f:
        pickle.dump(latent_dict, f)
