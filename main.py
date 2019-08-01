import numpy as np
import gym
import os
from gym.wrappers import ObservationWrapper
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, Model, Sequential, Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.optimizers import Adam
#
from rl.agents.dqn4hrl import DQNAgent4Hrl
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.agents.ddpg import DDPGAgent
from rl.processors import WhiteningNormalizerProcessor


tf.compat.v1.disable_eager_execution()

upper_nb_actions = 3
lower_nb_actions = 2
TIME_STEPS = 10
TBD_total = 56
TBD_left = 38
TBD_straight = 56
TBD_right = 38
LSTM_HIDDEN = 128
ENCODE_LSTM_HIDDEN = 64
ENV_NAME = 'EndtoendEnv-v0'
# Get the environment and extract the number of actions.
current_path = os.path.dirname(__file__)
env = gym.make(ENV_NAME, setting_path=current_path + '/rl/Scenario/Highway_endtoend/', plan_horizon=30, history_len=TIME_STEPS)
env = ObservationWrapper(env)
# np.random.seed(123)
# env.seed(123)
# nb_actions = env.action_space.n


def build_models():
    # build upper model.
    upper_model = Sequential(name='upper_model')
    upper_model.add(layers.LSTM(128, input_shape=(TIME_STEPS, TBD_total)))  # A 3D tensor [batch, timesteps, inputdim]
    upper_model.add(layers.Dense(upper_nb_actions, activation='relu'))

    # build lower actor shared part----------------------------------------------
    actor_lstm_model = Sequential(name='shared_actor_lstm_model')
    actor_lstm_model.add(layers.LSTM(LSTM_HIDDEN, input_shape=(TIME_STEPS, ENCODE_LSTM_HIDDEN)))

    dense_lstm_input = Input(shape=(LSTM_HIDDEN,))
    dense_indicator_input = Input(shape=(3,))
    dense_input = layers.concatenate([dense_lstm_input, dense_indicator_input])
    h = layers.Dense(32)(dense_input)
    # prob_output = layers.Dense(1, activation='sigmoid')(h)
    # vel_output = layers.Dense(1, activation='relu')(h)
    # out = layers.concatenate([prob_output, vel_output], axis=1)
    out = layers.Dense(2, activation='tanh')(h)
    actor_dense_model = Model(inputs=[dense_lstm_input, dense_indicator_input],
                              outputs=out, name='shared_actor_dense_model')

    # build lower actor left
    left_state_input = Input(shape=(TIME_STEPS, TBD_left + 3), name='state_left')
    lstm_input_left = Lambda(lambda x: x[:, :, :-3], output_shape=(TIME_STEPS, TBD_left))(left_state_input)
    indicator_input_left = Lambda(lambda x: x[:, 0, -3:], output_shape=(3,))(left_state_input)

    encoded_tensor = layers.LSTM(ENCODE_LSTM_HIDDEN, return_sequences=True, name='actor_left_encoder')(
        lstm_input_left)
    lstm_out = actor_lstm_model(encoded_tensor)  # reuse lstm_model
    out = actor_dense_model([lstm_out, indicator_input_left])  # reuse dense_model
    left_actor_model = Model(inputs=left_state_input,
                             outputs=out, name='left_actor_model')

    # build lower actor straight
    straight_state_input = Input(shape=(TIME_STEPS, TBD_straight + 3), name='state_straight')
    lstm_input_straight = Lambda(lambda x: x[:, :, :-3], output_shape=(TIME_STEPS, TBD_straight))(straight_state_input)
    indicator_input_straight = Lambda(lambda x: x[:, 0, -3:], output_shape=(3,))(straight_state_input)

    encoded_tensor = layers.LSTM(ENCODE_LSTM_HIDDEN, return_sequences=True, name='actor_straight_encoder')(
        lstm_input_straight)
    lstm_out = actor_lstm_model(encoded_tensor)  # reuse lstm_model
    out = actor_dense_model([lstm_out, indicator_input_straight])  # reuse dense_model
    straight_actor_model = Model(inputs=straight_state_input,
                                 outputs=out, name='straight_actor_model')

    # build lower actor right
    right_state_input = Input(shape=(TIME_STEPS, TBD_right + 3), name='state_right')
    lstm_input_right = Lambda(lambda x: x[:, :, :-3], output_shape=(TIME_STEPS, TBD_right))(right_state_input)
    indicator_input_right = Lambda(lambda x: x[:, 0, -3:], output_shape=(3,))(right_state_input)

    encoded_tensor = layers.LSTM(ENCODE_LSTM_HIDDEN, return_sequences=True, name='actor_right_encoder')(
        lstm_input_right)
    lstm_out = actor_lstm_model(encoded_tensor)  # reuse lstm_model
    out = actor_dense_model([lstm_out, indicator_input_right])  # reuse dense_model
    right_actor_model = Model(inputs=right_state_input,
                              outputs=out, name='right_actor_model')

    # build lower critic shared part--------------------------------------------
    critic_lstm_model = Sequential(name='shared_critic_lstm_model')
    critic_lstm_model.add(layers.LSTM(LSTM_HIDDEN, input_shape=(TIME_STEPS, ENCODE_LSTM_HIDDEN)))

    dense_lstm_input = Input(shape=(LSTM_HIDDEN,))
    dense_action_input = Input(shape=(2,))
    dense_indicator_input = Input(shape=(3,))
    dense_input = layers.concatenate([dense_lstm_input, dense_action_input, dense_indicator_input])
    h = layers.Dense(32)(dense_input)
    q_output = layers.Dense(1, activation='relu')(h)
    critic_dense_model = Model(inputs=[dense_lstm_input, dense_action_input, dense_indicator_input],
                                        outputs=q_output, name='shared_critic_dense_model')

    # build lower critic left
    action_input = Input(shape=(2,), name='action')

    encoded_tensor = layers.LSTM(ENCODE_LSTM_HIDDEN, return_sequences=True, name='critic_left_encoder')(
        lstm_input_left)
    lstm_out = critic_lstm_model(encoded_tensor)  # reuse lstm_model
    q_output = critic_dense_model([lstm_out, action_input, indicator_input_left])  # reuse dense_model
    left_critic_model = Model(inputs=[left_state_input, action_input],
                              outputs=q_output, name='left_critic_model')

    # build lower critic straight
    encoded_tensor = layers.LSTM(ENCODE_LSTM_HIDDEN, return_sequences=True, name='critic_straight_encoder')(
        lstm_input_straight)
    lstm_out = critic_lstm_model(encoded_tensor)  # reuse lstm_model
    q_output = critic_dense_model([lstm_out, action_input, indicator_input_straight])  # reuse dense_model
    straight_critic_model = Model(inputs=[straight_state_input, action_input],
                                  outputs=q_output, name='straight_critic_model')

    # build lower critic right
    encoded_tensor = layers.LSTM(ENCODE_LSTM_HIDDEN, return_sequences=True, name='critic_right_encoder')(
        lstm_input_right)
    lstm_out = critic_lstm_model(encoded_tensor)  # reuse lstm_model
    q_output = critic_dense_model([lstm_out, action_input, indicator_input_right])  # reuse dense_model
    right_critic_model = Model(inputs=[right_state_input, action_input],
                               outputs=q_output, name='right_critic_model')
    model_dict = dict(upper_model=upper_model,
                      left_actor_model=left_actor_model,
                      left_critic_model=left_critic_model,
                      straight_actor_model=straight_actor_model,
                      straight_critic_model=straight_critic_model,
                      right_actor_model=right_actor_model,
                      right_critic_model=right_critic_model)
    return model_dict, action_input
def action_fn():
    upper_action = np.random.choice([0, 1, 2])
    delta_x_norm = (np.random.random() - 0.5) * 2
    acc_norm = (np.random.random() - 0.5) * 2

    return upper_action, (delta_x_norm, acc_norm)

model_dict, critic_action_input = build_models()
upper_model = model_dict['upper_model']

left_actor_model = model_dict['left_actor_model']
left_critic_model = model_dict['left_critic_model']

straight_actor_model = model_dict['straight_actor_model']
straight_critic_model = model_dict['straight_critic_model']

right_actor_model = model_dict['right_actor_model']
right_critic_model = model_dict['right_critic_model']

# print(left_actor_model.input, left_critic_model.input)
# print(right_actor_model.input, right_critic_model.input)
# print(left_critic_model.input.index(critic_action_input))
# import json
# import pprint
# pprint.pprint(json.loads(left_actor_model.to_json()))
# pprint.pprint(json.loads(left_critic_model.to_json()))



#
# tf.keras.utils.plot_model(left_actor_model, 'left_actor_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(straight_actor_model, 'straight_actor_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(right_actor_model, 'right_actor_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(left_critic_model, 'left_critic_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(straight_critic_model, 'straight_critic_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(right_critic_model, 'right_critic_model_with_shape_info.png', show_shapes=True)


# define hyperparameter for DDPG agent
MEMORY_LIMIT = 100000
WINDOW_LENGTH = 1
NB_STEPS_WARMUP_CRITIC = 40
NB_STEPS_WARMUP_ACTOR = 40
GAMMA = 0.99
TARGET_MODEL_UPDATE = 1e-3
RANDOM_PROCESS_THETA = 0.15
RANDOM_PROCESS_MU = 0.
RANDOM_PROCESS_SIGMA = 0.3
OPTIMIZER_LR = 0.001
OPTIMIZER_CLIPNORM = 1.0

# define hyperparameter for DQN4Hrl
MEMORY_LIMIT_UPPER = 50000
WINDOW_LENGTH_UPPER = 1
NB_STEPS_WARMUP_STEP = 100
TARGET_MODEL_UPDATE_UPPER = 1e-2
OPTIMIZER_LR_UPPER = 0.001


# turn left agent
left_processor = WhiteningNormalizerProcessor()
left_memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGTH)
left_random_process = OrnsteinUhlenbeckProcess(size=lower_nb_actions, theta=RANDOM_PROCESS_THETA, mu=RANDOM_PROCESS_MU, sigma=RANDOM_PROCESS_SIGMA)
left_agent = DDPGAgent(processor=left_processor, nb_actions=lower_nb_actions, actor=left_actor_model,
                       critic=left_critic_model, critic_action_input=critic_action_input,
                       memory=left_memory, nb_steps_warmup_critic=NB_STEPS_WARMUP_CRITIC, nb_steps_warmup_actor=NB_STEPS_WARMUP_ACTOR,
                       random_process=left_random_process, gamma=GAMMA, target_model_update=TARGET_MODEL_UPDATE)
left_agent.compile(Adam(lr=OPTIMIZER_LR, clipnorm=OPTIMIZER_CLIPNORM), metrics=['mae'])

# go straight agent
straight_processor = WhiteningNormalizerProcessor()
straight_memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGTH)
straight_random_process = OrnsteinUhlenbeckProcess(size=lower_nb_actions, theta=RANDOM_PROCESS_THETA, mu=RANDOM_PROCESS_MU, sigma=RANDOM_PROCESS_SIGMA)
straight_agent = DDPGAgent(processor=straight_processor, nb_actions=lower_nb_actions, actor=straight_actor_model,
                           critic=straight_critic_model, critic_action_input=critic_action_input,
                           memory=straight_memory, nb_steps_warmup_critic=NB_STEPS_WARMUP_CRITIC, nb_steps_warmup_actor=NB_STEPS_WARMUP_ACTOR,
                           random_process=straight_random_process, gamma=GAMMA, target_model_update=TARGET_MODEL_UPDATE)
straight_agent.compile(Adam(lr=OPTIMIZER_LR, clipnorm=OPTIMIZER_CLIPNORM), metrics=['mae'])

# turn right agent
right_processor = WhiteningNormalizerProcessor()
right_memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGTH)
right_random_process = OrnsteinUhlenbeckProcess(size=lower_nb_actions, theta=RANDOM_PROCESS_THETA, mu=RANDOM_PROCESS_MU, sigma=RANDOM_PROCESS_SIGMA)
right_agent = DDPGAgent(processor=right_processor, nb_actions=lower_nb_actions, actor=right_actor_model, critic=right_critic_model,
                        critic_action_input=critic_action_input,
                        memory=right_memory, nb_steps_warmup_critic=NB_STEPS_WARMUP_CRITIC, nb_steps_warmup_actor=NB_STEPS_WARMUP_ACTOR,
                        random_process=right_random_process, gamma=GAMMA, target_model_update=TARGET_MODEL_UPDATE)
right_agent.compile(Adam(lr=OPTIMIZER_LR, clipnorm=OPTIMIZER_CLIPNORM), metrics=['mae'])

processor = WhiteningNormalizerProcessor()
memory = SequentialMemory(limit=MEMORY_LIMIT_UPPER, window_length=WINDOW_LENGTH_UPPER)
policy = BoltzmannQPolicy()
dqn = DQNAgent4Hrl(processor=processor, model=upper_model, turn_left_agent=left_agent, go_straight_agent=straight_agent,
                   turn_right_agent=right_agent, nb_actions=upper_nb_actions, memory=memory, nb_steps_warmup=NB_STEPS_WARMUP_STEP,
                   target_model_update=TARGET_MODEL_UPDATE_UPPER, policy=policy, enable_double_dqn=True)
dqn.compile(Adam(lr=OPTIMIZER_LR_UPPER), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit_hrl(env, nb_steps=50000, visualize=False, verbose=2, random_start_step_policy=action_fn, save_interval=100)

# # After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#
# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)
