import os.path
import subprocess
import tempfile
import warnings
from functools import reduce
from typing import Tuple, NamedTuple, List, Union

import h5py
import numpy as np
from PIL import Image
from keras.models import Model, Sequential
from keras.layers import Layer
from keras import optimizers
# noinspection PyPep8Naming
from keras import backend as K


class ConvOp(NamedTuple):
    kernel_size: int
    strides: int


def divisible_padding(dim: int, ops: List[ConvOp]):

    rough_final_dim = dim
    for op in ops:
        rough_final_dim = (rough_final_dim - op.kernel_size) / op.strides + 1
    final_dim = int(np.ceil(rough_final_dim))
    start_dim = final_dim
    for op in reversed(ops):
        start_dim = (start_dim - 1) * op.strides + op.kernel_size
    total_padding = int(start_dim - dim)
    if total_padding % 2 == 0:
        before = after = total_padding // 2
    else:
        before = total_padding // 2
        after = total_padding - before
    return before, after


ImagePadding = Tuple[Tuple[int, int], Tuple[int, int]]
ImageShape = Tuple[int, int]


def calc_optimal_convnet_padding(
        orig_image_size: Tuple[int, int],
        conv_ops: List[ConvOp])->Tuple[ImagePadding, ImageShape]:
    """
    Finds out how to pad observations to make their size evenly divisible
    by given convolution operations. This guarantees that the original
    image could then be reconstructed back using transpose convolutions.
    The function assumes that all convolutions use "valid" padding.
    If you have convolutions with "same" padding, just omit those operations
    in the conv_ops list.
    """
    pad_width = (
        divisible_padding(orig_image_size[0], conv_ops),
        divisible_padding(orig_image_size[1], conv_ops))
    adjusted_shape = (
        orig_image_size[0] + pad_width[0][1] + pad_width[0][1],
        orig_image_size[1] + pad_width[1][1] + pad_width[1][1])
    return pad_width, adjusted_shape


class OneHotConverter:

    def __init__(self, num_categories: int):
        assert num_categories > 0
        self.num_categories = num_categories
        self._all_options = np.eye(num_categories)

    def __call__(self, value: np.ndarray):
        return self._all_options[value]


def save_optimizer_weights(model: [Model, Sequential],
                           model_group: h5py.Group):
    """
    Saves optimizer`s weights in a particular section of HDF5 file,
    which allows for storing multiple models and their optimizers in a single
    file.
    The code has been mostly taken from the guts of the Keras itself.
    """
    optimizer = getattr(model, 'optimizer', None)  # type: optimizers.Optimizer
    if optimizer is None:
        warnings.warn('Cannot save optimizer state for model {} because '
                      'it does not have an optimizer'.format(model.name))
        return
    symbolic_weights = getattr(optimizer, 'weights')
    if symbolic_weights:
        optimizer_weights_group = model_group.create_group('optimizer_weights')
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights,
                                         weight_values)):
            # Default values of symbolic_weights is /variable
            # for Theano and CNTK
            if K.backend() == 'theano' or K.backend() == 'cntk':
                if hasattr(w, 'name'):
                    if w.name.split('/')[-1] == 'variable':
                        name = str(w.name) + '_' + str(i)
                    else:
                        name = str(w.name)
                else:
                    name = 'param_' + str(i)
            else:
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))
        optimizer_weights_group.attrs['weight_names'] = weight_names
        for name, val in zip(weight_names, weight_values):
            param_dataset = optimizer_weights_group.create_dataset(
                name,
                val.shape,
                dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dataset[()] = val
            else:
                param_dataset[:] = val


def load_optimizer_weights(model: Union[Model, Sequential],
                           model_group: h5py.Group):
    """
    Loads optimizer`s weights from a particular section of HDF5 file,
    which allows for storing multiple models and their optimizers in a single
    file.
    """
    optimizer = getattr(model, 'optimizer', None)  # type: optimizers.Optimizer
    if optimizer is None:
        warnings.warn('Cannot save optimizer state for model {} because '
                      'it does not have an optimizer'.format(model.name))
        return
    if 'optimizer_weights' in model_group:
        # Build train function (to get weight updates).
        if isinstance(model, Sequential):
            # noinspection PyProtectedMember
            model.model._make_train_function()
        else:
            # noinspection PyProtectedMember
            model._make_train_function()
        optimizer_weights_group = model_group['optimizer_weights']
        optimizer_weight_names = [
            n.decode('utf8')
            for n in optimizer_weights_group.attrs['weight_names']]
        optimizer_weight_values = [
            optimizer_weights_group[n] for n in optimizer_weight_names]
        try:
            model.optimizer.set_weights(optimizer_weight_values)
        except ValueError:
            warnings.warn('Error in loading the saved optimizer '
                          'state. As a result, your model is '
                          'starting with a freshly initialized '
                          'optimizer.')


class PopArtLayer(Layer):
    """
    Automatic network output scale adjuster, which is supposed to keep
    the output of the network up to date as we keep updating moving
    average and variance of discounted returns.

    Part of the PopArt algorithm described in DeepMind's paper
    "Learning values across many orders of magnitude"
    (https://arxiv.org/abs/1602.07714)
    """
    def __init__(self, beta=1e-4, epsilon=1e-4, stable_rate=0.1,
                 min_steps=1000, **kwargs):
        """
        :param beta: a value in range (0..1) controlling sensitivity to changes
        :param epsilon: a minimal possible value replacing standard deviation
                if the original one is zero.
        :param stable_rate: Pop-part of the algorithm will kick in only when
            the amplitude of changes in standard deviation will drop
            to this value (stabilizes). This protects pop-adjustments from
            being activated too soon, which would lead to weird values
            of `W` and `b` and numerical instability.
        :param min_steps: Minimal number of steps before it even begins
            possible for Pop-part to become activated (an extra precaution
            in addition to the `stable_rate`).
        :param kwargs: any extra Keras layer parameters, like name, etc.
        """
        self.beta = beta
        self.epsilon = epsilon
        self.stable_rate = stable_rate
        self.min_steps = min_steps
        super().__init__(**kwargs)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=(), dtype='float32',
            initializer='ones', trainable=False)
        self.bias = self.add_weight(
            name='bias', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.mean = self.add_weight(
            name='mean', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.mean_of_square = self.add_weight(
            name='mean_of_square', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.step = self.add_weight(
            name='step', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.pop_is_active = self.add_weight(
            name='pop_is_active', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.kernel * inputs + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

    def de_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Converts previously normalized data into original values.
        """
        online_mean, online_mean_of_square = K.batch_get_value(
            [self.mean, self.mean_of_square])
        std_dev = np.sqrt(online_mean_of_square - np.square(online_mean))
        return (x * (std_dev if std_dev > 0 else self.epsilon)
                + online_mean)

    def pop_art_update(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Performs ART (Adaptively Rescaling Targets) update,
        adjusting normalization parameters with respect to new targets x.
        Updates running mean, mean of squares and returns
        new mean and standard deviation for later use.
        """
        assert len(x.shape) == 2, "Must be 2D (batch_size, time_steps)"
        beta = self.beta
        (old_kernel, old_bias, old_online_mean,
         old_online_mean_of_square, step, pop_is_active) = K.batch_get_value(
            [self.kernel, self.bias, self.mean,
             self.mean_of_square, self.step, self.pop_is_active])

        def update_rule(old, new):
            """
            Update rule for running estimations,
            dynamically adjusting sensitivity with every time step
            to new data (see Eq. 10 in the paper).
            """
            nonlocal step
            step += 1
            adj_beta = beta / (1 - (1 - beta)**step)
            return (1 - adj_beta) * old + adj_beta * new

        x_means = np.stack([x.mean(axis=0), np.square(x).mean(axis=0)], axis=1)
        # Updating normalization parameters (for ART)
        online_mean, online_mean_of_square = reduce(
            update_rule, x_means,
            np.array([old_online_mean, old_online_mean_of_square]))
        old_std_dev = np.sqrt(
            old_online_mean_of_square - np.square(old_online_mean))
        std_dev = np.sqrt(online_mean_of_square - np.square(online_mean))
        old_std_dev = old_std_dev if old_std_dev > 0 else std_dev
        # Performing POP (Preserve the Output Precisely) update
        # but only if we are not in the beginning of the training
        # when both mean and std_dev are close to zero or still
        # stabilizing. Otherwise POP kernel (W) and bias (b) can
        # become very large and cause numerical instability.
        std_is_stable = (
            step > self.min_steps
            and np.abs(1 - old_std_dev / std_dev) < self.stable_rate)
        if (int(pop_is_active) == 1 or
                (std_dev > self.epsilon and std_is_stable)):
            new_kernel = old_std_dev * old_kernel / std_dev
            new_bias = (
                (old_std_dev * old_bias + old_online_mean - online_mean)
                / std_dev)
            pop_is_active = 1
        else:
            new_kernel, new_bias = old_kernel, old_bias
        # Saving updated parameters into graph variables
        var_update = [
            (self.kernel, new_kernel),
            (self.bias, new_bias),
            (self.mean, online_mean),
            (self.mean_of_square, online_mean_of_square),
            (self.step, step),
            (self.pop_is_active, pop_is_active)]
        K.batch_set_value(var_update)
        return online_mean, std_dev

    def update_and_normalize(self, x: np.ndarray) -> Tuple[np.ndarray,
                                                           float, float]:
        """
        Normalizes given tensor `x` and updates parameters associated
        with PopArt: running means (art) and network's output scaling (pop).
        """
        mean, std_dev = self.pop_art_update(x)
        result = ((x - mean) / (std_dev if std_dev > 0 else self.epsilon))
        return result, mean, std_dev


class MovieWriter:
    """
    Helps to record the gameplay as a video or gif-animation.
    """
    fps = 30
    frame_max_digits = 6

    def __init__(self, output_file_name: str, movie_width=640):
        self.image_directory = tempfile.TemporaryDirectory(
            prefix='kerl_recording')
        self.movie_width = movie_width
        self.frame_counter = 0
        self.output_file_name = output_file_name
        self.frame_file_format = 'frame%0{}d.png'.format(self.frame_max_digits)

    def save_frame(self, picture: np.ndarray):
        image = Image.fromarray(picture, 'RGB')
        self.frame_file_name(self.frame_counter)
        image.save(
            self.frame_file_name(self.frame_counter))
        self.frame_counter += 1

    def frame_file_name(self, frame_idx: int):
        return os.path.join(
            self.image_directory.name,
            self.frame_file_format % frame_idx)

    def finish(self):
        print('Assembling video from frames...', end='', flush=True)
        subprocess.run([
            'ffmpeg',
            '-i', os.path.join(self.image_directory.name,
                               self.frame_file_format),
            '-r', str(self.fps), '-pix_fmt', 'yuv420p', '-y',
            '-vf', 'scale={}:-1'.format(self.movie_width),
            '-loglevel', 'error',
            self.output_file_name])
        self.image_directory.cleanup()
        print('Done', flush=True)


class SamplingLayer(Layer):

    def call(self, args, **kwargs):
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=K.shape(z_mean),
            mean=0., stddev=1.0)
        sample = z_mean + K.exp(z_log_var / 2) * epsilon
        return sample

    def compute_output_shape(self, input_shape):
        return input_shape[0]
