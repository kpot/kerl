World Models
============
This is an attempt to reproduce an interesting paper ["World Models"](https://arxiv.org/abs/1803.10122)
written by David Ha and Jürgen Schmidhuber.

The paper offers a fresh look on how can we train RL agents by essentially
splitting them into two parts:

1. The RL policy itself, which needs to be trained using reinforcement learning
   algorithms, which are currently known to be far from perfect.
2. The model of reality, which can be very effectively trained from observations
   in an unsupervised manner, using well proven and robust deep learning methods.

The premise here is that the policy can make use of high-level features learned
by the model, and thus should learn faster, and allow usage of a broad variety
training of methods, including, among others,
[genetic algorithms](https://arxiv.org/abs/1712.06567) and
[evolution strategies](https://blog.openai.com/evolution-strategies/).

The authors don't stop there and explore many intriguing possibilities such
as using the learned model for "dreaming" (hallucinating), and even training
the agent completely inside such dream.

This code tries to reproduce the papers's architecture with few minor tweaks:

1. For variational autoencoder ReLU activation has been replaced with
   [SELU](https://arxiv.org/abs/1706.02515) since it demonstrated faster
   training speed without noticeable change in quality.
2. Autoencoder works with unscaled observations and uses one more layer both
   for encoding and decoding parts.
3. The controller relies on A2C algorithm instead of CMA-ES used
   in the paper. It has two hidden dense layers and two output layers
   corresponding to the policy and the value function.

Experiments
-----------

Two versions have been tested playing Atari Ms. Pacman: with latent space vector
*z* having size of 128 and 512 dimensions. Also all networks were trained
simultaneously, without pre-training VAE on random observations.

Here's video demonstrating the result of training with *z &isin; R<sup>​128</sup>*
and the RNN (LSTM) having 512 units, performed using following command:

    python -m kerl.wm.run --model-path model-MsPacman-v0-WM.h5 train --num-envs 16 --lr 2e-4 --time-horizon 5 --gym-env MsPacman-v0 --reward-scale 0.02 --latent-dim 128 --world-rnn-units 512


[![WM agent playing Ms.Pacman](https://img.youtube.com/vi/IR_srJfJzco/0.jpg)](https://www.youtube.com/watch?v=IR_srJfJzco)

Increasing dimensionality of *z* from 128 to 512 as well as the number of LSTM
units from 512 to 1024 does not improve the performance. Both versions
reach average score of about 1250 in MsPacman after about 30M iterations
and do not improve any further during the next 30M steps of training.
This isn't a great score for Pacman, the game where many model-free algorithms
easily achieve score above 3000.

Perhaps such poor results can be explained by the level of "noise"
in observations over time, caused by VAE's imperfections in encoding.
For example, VAE sometimes can turn Ms. Pacman into a ghost, or "resurrect"
some of the pellets, thus confusing the agent. The environment model obediently
learns to repeat those errors, confusing everything even further.
