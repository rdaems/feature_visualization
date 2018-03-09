# feature visualization
# https://distill.pub/2017/feature-visualization/

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform
from skimage.transform import resize
from skimage.io import imsave
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_transformation_matrix():
    # values from https://distill.pub/2017/feature-visualization/#d-footnote-9
    j1 = (np.random.randint(-16, 17), np.random.randint(-16, 17))
    s = np.random.choice([1, 0.975, 1.025, 0.95, 1.05])
    r = np.random.randint(-5, 6)
    j2 = (np.random.randint(-8, 9), np.random.randint(-8, 9))

    # # a bit less heavy
    # j1 = (np.random.randint(-4, 7), np.random.randint(-4, 7))
    # s = np.random.choice([1, 0.975, 1.025])
    # r = np.random.randint(-5, 6)
    # j2 = (np.random.randint(-2, 3), np.random.randint(-2, 3))

    J1 = np.eye(3)
    J1[0:2, 2] = j1
    S = np.eye(3)
    S[0, 0] = s
    S[1, 1] = s
    R = np.eye(3)
    R[0:2, 0:2] = [[np.cos(np.radians(r)), -np.sin(np.radians(r))], [np.sin(np.radians(r)), np.cos(np.radians(r))]]
    J2 = np.eye(3)
    J2[0:2, 2] = j2
    T = J2 @ R @ S @ J1
    return T


def transform_image(image, T, origin='center', inverse=False):
    # image: 2 or 3 dimensional image (h, w, c)
    # T: 3x3 augmented transformation matrix.
    # origin: 'center' or 'topleft': where is the origin of the T transformation.
    # inverse: T is defined as the transformation from the output to the input (invers).
    # If inverse is False, T is inversed.
    # now the transformation is from the input to the output
    if image.ndim == 3:
        # do the transformation for every channel
        num_channels = image.shape[2]
        transformed_channels = []
        for i in range(num_channels):
            transformed_channels.append(transform_image(image[..., i], T, origin=origin, inverse=inverse))
        return np.stack(transformed_channels, axis=-1)
    if image.ndim != 2:
        raise ValueError('Wrong number of dimensions.')

    h, w = image.shape

    if not inverse:
        T_i = np.linalg.inv(T)
    else:
        T_i = T

    if origin == 'center':
        c = (h / 2, w / 2)
    else:
        c = (0, 0)

    T_o = [[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]] @ T_i @ [[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]]

    transformed = affine_transform(image, T_o, mode='constant')
    return transformed


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def process_gradient(g):
    return g


def visualize(model, layer_name, filter_index, img_width=128, img_height=128, step=1., num_iter=20, num_octaves=1, octave_scale=1):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads = normalize(grads)
    iterate = K.function([model.input, K.learning_phase()], [loss, grads])

    # multiscale setup
    octaves = octave_scale ** - np.arange(num_octaves)[::-1]
    img_height = np.round(img_height * octaves).astype(np.int)
    img_width = np.round(img_width * octaves).astype(np.int)

    # we start from a gray image with some random noise
    x = np.random.random((img_height[0], img_width[0], 3))
    x = (x - 0.5) * 6

    for o in range(num_octaves):
        x = resize(x, (img_height[o], img_width[o]))
        for i in range(num_iter):
            # affine transformation
            T = get_transformation_matrix()

            x_t = transform_image(x, T)

            loss_value, grads_value = iterate([x_t[None, ...], 0])

            # inverse transform gradient
            grads_value_transformed = transform_image(grads_value[0], T, inverse=True)

            grads_value_transformed = process_gradient(grads_value_transformed)

            x += grads_value_transformed * step

            imsave('debug/%d_%d.bmp' % (o, i), deprocess_image(x))

            print('Current loss value:', loss_value)

    # deprocess the resulting input image
    img = deprocess_image(x)
    return img


if __name__ == '__main__':
    from keras.applications import vgg16

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    model.summary()
    layer_name = 'block5_conv2'
    filter_id = 74
    img_width = 128
    img_height = 128
    img = visualize(model, layer_name, filter_id, img_width=img_width, img_height=img_height, step=1., num_iter=20, num_octaves=3, octave_scale=1.4)

    imsave('%s_%d_%dx%d.bmp' % (layer_name, filter_id, img_width, img_height), img)

    plt.figure()
    plt.imshow(img)
    plt.show()
