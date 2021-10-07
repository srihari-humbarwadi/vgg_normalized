import os

import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from tqdm import tqdm

flags.DEFINE_string(
    name='image_dir',
    default='Imagenet2012/ILSVRC2012_img_val',
    help='directory containing images')

flags.DEFINE_string(
    name='model_name',
    default='vgg19',
    help='vgg16|vgg19')

flags.DEFINE_string(
    name='output_weights_path',
    default=None,
    required=True,
    help='path to store the rescaled weights')

FLAGS = flags.FLAGS

_model_fns = {
    'vgg16': tf.keras.applications.VGG16,
    'vgg19': tf.keras.applications.VGG19,
}


def preprocess_fn(path, resize_to=(224, 224)):
    image = tf.image.decode_image(tf.io.read_file(path), channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, size=resize_to)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image


def get_dataset(file_pattern):
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _get_channelwise_mean(tensor):
    return tf.reduce_mean(tensor, axis=[0, 1, 2])


def main(_):
    # There is no need to enforce h5 format. But it makes it easier to plug in
    # rescaled weights into existing codes bases that use keras applications.
    if not FLAGS.output_weights_path.endswith('.h5'):
        raise ValueError('Only h5 format is supported')

    dataset = get_dataset(os.path.join(FLAGS.image_dir, '*'))
    logging.info('Created image dataset with {} images'.format(len(dataset)))

    model = _model_fns[FLAGS.model_name](include_top=False, weights='imagenet')
    mean_activations = {}

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            mean_activations[layer.name] = _get_channelwise_mean(layer.output)

    mean_activations_model = tf.keras.Model(model.input, mean_activations)
    predict_fn = tf.function(
        lambda x: mean_activations_model(x, training=False))

    logging.info('Accumulating layerwise channel means for all images')
    num_images_seen = 0
    activations = {
        k: tf.zeros(shape=v.shape)
        for k, v in mean_activations_model.output.items()
    }
    for image in tqdm(dataset):
        current_activations = predict_fn(image)

        for k, v in current_activations.items():
            activations[k] += v

        num_images_seen += 1

    for k, v in activations.items():
        activations[k] = v / num_images_seen

    logging.info('Rescaling weights')
    previous_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            current_activation_means = activations[layer.name].numpy()
            kernel = layer.kernel.numpy()
            bias = layer.bias.numpy()

            if previous_conv_layer is not None:
                previous_activation_means = activations[previous_conv_layer]
                assert previous_activation_means.shape[0] == kernel.shape[2]
                kernel *= np.reshape(previous_activation_means, [1, 1, -1, 1])

            kernel /= np.reshape(current_activation_means, [1, 1, 1, -1])
            bias /= current_activation_means

            layer.kernel.assign(kernel)
            layer.bias.assign(bias)
            previous_conv_layer = layer.name


    model.save_weights(FLAGS.output_weights_path)
    logging.info('Saved rescaled weights to {}'.format(
        FLAGS.output_weights_path))


if __name__ == '__main__':
    app.run(main)
