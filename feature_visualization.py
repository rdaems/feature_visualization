# feature visualization
# https://distill.pub/2017/feature-visualization/

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
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


def visualize(model, layer_name, filter_index, img_width=128, img_height=128, step=1., num_iter=20):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]

    grads = normalize(grads)

    # we start from a gray image with some random noise
    input_img_data = np.random.random((img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 6

    iterate = K.function([model.input, K.learning_phase()], [loss, grads])


    for i in range(num_iter):
        # affine transformation
        T = get_transformation_matrix()

        input_img_data_transformed = transform_image(input_img_data, T)

        loss_value, grads_value = iterate([input_img_data_transformed[None, ...], 0])

        # inverse transform gradient
        grads_value_transformed = transform_image(grads_value[0], T, inverse=True)

        input_img_data += grads_value_transformed * step

        print('Current loss value:', loss_value)

    # deprocess the resulting input image
    img = deprocess_image(input_img_data)
    return img


if __name__ == '__main__':
    from keras.applications import vgg16

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    img = visualize(model, 'block4_conv2', 7, img_width=128, img_height=128, step=1., num_iter=20)

    plt.figure()
    plt.imshow(img)
    plt.show()
