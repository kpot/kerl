KERL
====
KERL is a collection of various Reinforcement Learning algorithms and related
techniques implemented purely using Keras.

The goal of the project is to create implementations of
state-of-the-art RL algorithms as well as a platform for developing
and testing new ones, yet keep the code simple and portable thanks
to Keras and its ability to use various backends.

This makes KERL very similar to
[OpenAI Baselines](https://github.com/openai/baselines), only with focus on Keras.

With KERL you can quickly train various agents to play Atari games from pixels
and dive into details of their implementation. Here's an example of such agent
trained with KERL:

[![Watch A2C agent playing Ms.Pacman](https://img.youtube.com/vi/odMpY3ogbUE/0.jpg)](http://www.youtube.com/watch?v=odMpY3ogbUE)

What works
----------

* [Advantage Actor-Critic (A2C)](./kerl/a2c) [original [paper](https://arxiv.org/abs/1602.01783)]
* [Proximal Policy Optimization (PPO)](./kerl/ppo) [original [paper](https://arxiv.org/abs/1707.06347)]
* [World Models (WM)](./kerl/wm) [original [paper](https://arxiv.org/abs/1803.10122)]

(click on the left link for more details about each implementation)

All algorithms support adaptive normalization of returns Pop-Art, described
in DeepMind's paper ["Learning values across many orders of magnitude"](https://arxiv.org/abs/1602.07714).
This greatly simplifies the training, often making it possible to just throw
the algorithm at a task and get a decent result.

This is also possible to run training on downscaled version of the game,
which can significantly increase the speed of training, although can
also damage the algorithms's performance depending on the resolution
you choose.

Current limitations
-------------------
Currently KERL does not support continuous control tasks and so far was tested
only on various Atari games supported by
[The Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
via OpenAI Gym.

Quick start
------------
Assuming you use some kind of Linux/MacOS and you have both Git and
Python >= 3.6 already installed and the python available as `python3`.
Here's how you can clone the repository and install the necessary libraries:

    git clone https://github.com/kpot/kerl.git
    cd kerl
    python3 -m venv kerl-sandbox
    source kerl-sandbox/bin/activate
    pip install -r requirements.txt

If you have difficulties finding Python 3.6 for your OS, perhaps
[pyenv](https://github.com/pyenv/pyenv) can help you.

You will also need to install some backend for Keras. For this example we'll
install TensorFlow.

    pip install tensorflow

Now you can start training your agent by calling

    python -m kerl.ppo.run --model-path model-MsPacman-v0-PPO-CNN.h5 train \
       --gym-env MsPacman-v0 --num-envs 16 --network CNN --lr 2e-4 \
       --time-horizon 5 --reward-scale 0.05
       
This command will train a PPO agent to play MsPacman by running 16 parallel
simulations, each of which will be run during 5 time steps for each
training cycle. The agent will use already classical convolutional neural
network with dense policy and value outputs, learning rate 2e-4 and scale the rewards
down by multiplying them to 0.05 (1/20) to avoid numerical instabilities.

The command will produce two outputs:

1. The agent's weights,
   regularly stored using using [HDF5 format](http://docs.h5py.org/en/latest/)
   in a file called `model-MsPacman-v0-PPO-CNN.h5`.
2. A file by default named `model-MsPacman-v0-PPO-CNN_history.txt`, containing
   records of all episodes played during the training. The exact format of this
   file is described [later](#history-file).

You can stop the training at any moment and then start it again, it will pick
up the progress from the last save.

If you need to train an agent with a different algoritm, you need to run
a differen module, like `python -m kerl.a2c.run ...`,
or `python -m kerl.wm.run ...`. Each module has some unique options to
tune its hyper-parameters, you can check them out by calling something like

    python -m kerl.<module>.run --model-path <model file name>.h5 train --help
    
Watch the agent playing
-----------------------
As the training progresses, you can run the agent and see how well it
has mastered the game so far.

Following the example from the [quick start](#quick-start), just run

    python -m kerl.ppo.run --model-path model-MsPacman-v0-PPO-CNN.h5 play

and you'll see the agent playing one episode.

Instead of watching the agent personally, you can record a video of the gameplay:


    python -m kerl.ppo.run --model-path model-MsPacman-v0-PPO-CNN.h5 play --record pacman_gameplay.mp4

Such videos are very useful for further analysis. Using cron and `--record`
option you can also schedule regular recording of such videos to better track
the agent's progress over time (you will need to have ffmpeg installed).

Check all available options by running

    python -m kerl.ppo.run --model-path model-MsPacman-v0-PPO-CNN.h5 play --help

History file
------------
a short snippet plotting the history (you will need to install matplotlib
using `pip install matplotlib` to make it work):

    import matplotlib.pyplot as plt
    from kerl.common.history import read_history
    history = read_history('model-MsPacman-v0-PPO-CNN_history.txt')
    plt.plot(history[:, 1], history[:, 0])
    plt.show()
 
History file is a simple text file with tab-separated columns,
[described in HistoryRecord class](./kerl/common/history.py). Just check
the source code comments describing each column.

Other details
-------------
* KERL does not use reward clipping, so popular in many papers. Such clipping
  can significantly change the agent's performance for the worse, completely
  changing the goals it pursues. Instead, KERL implements reward scaling
  and adaptive Pop-Art normalization which can be used independently or together.
  Be aware that PPO due to its usage of
  clipping within its loss function, can still be prone to the same behavioral
  changes. Because of that, more "primitive" A2C, although being less sample
  efficient, can sometimes achieve much greater score given enough time.
* Some architectures (such as [World Models](https://arxiv.org/abs/1803.10122))
  may employ multiple Keras models working together. All of them will be
  stored in separate sections of the same HDF5 file. Which means if you need
  to extract and reuse some part of the model, you will have to do it yourself,
  although this should not be difficult.

Similar projects
----------------
You may also want to take a look at these projects

* [Deep Reinforcement Learning for Keras](https://github.com/keras-rl/keras-rl):
  a good collection of Deep RL algorithms implemented in Keras.
  Mostly variations of Deep Q-learning.
* [OpenAI Baselines](https://github.com/openai/baselines): many of the most
  common RL algorithms to the day, implemented in TensorFlow.
