# Inspired from OpenAI Baselines

import gym
import numpy as np
import random

def set_global_seeds(i):
    try:
        import tensorflow as tf
        tf.random.set_seed(i)
    except ImportError:
        pass
    np.random.seed(i)
    random.seed(i)
