import numpy as np
import gym
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, Model, Sequential, Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.optimizers import Adam
#
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# ENV_NAME = 'AutonomousDriving-v0'
#
#
# # Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
# nb_actions = env.action_space.n
nb_actions = 3
TIME_STEPS = 10
TBD_total = 7
TBD_left = 7
TBD_straight = 7
TBD_right = 7
LSTM_HIDDEN = 128

def build_models():
    # build upper model.
    upper_model = Sequential(name='upper_model')
    upper_model.add(layers.LSTM(128))  # A 3D tensor [batch, timesteps, inputdim]
    upper_model.add(layers.Dense(nb_actions, activation='relu'))

    # build lower actor shared part----------------------------------------------
    actor_lstm_model = Sequential(name='shared_actor_lstm_model')
    actor_lstm_model.add(layers.LSTM(LSTM_HIDDEN))

    dense_lstm_input = Input(shape=(LSTM_HIDDEN,))
    dense_indicator_input = Input(shape=(3,))
    dense_input = layers.concatenate([dense_lstm_input, dense_indicator_input])
    h = layers.Dense(32)(dense_input)
    prob_output = layers.Dense(1, activation='sigmoid')(h)
    vel_output = layers.Dense(1, activation='relu')(h)
    out = layers.concatenate([prob_output, vel_output], axis=1)
    actor_dense_model = Model(inputs=[dense_lstm_input, dense_indicator_input],
                              outputs=out, name='shared_actor_dense_model')

    # build lower actor left
    left_state_input = Input(shape=(TIME_STEPS, TBD_left + 3), name='state_left')
    lstm_input_left = Lambda(lambda x: x[:, :, :-3], output_shape=(TIME_STEPS, TBD_left))(left_state_input)
    indicator_input_left = Lambda(lambda x: x[:, 0, -3:], output_shape=(3,))(left_state_input)

    encoded_tensor = layers.LSTM(LSTM_HIDDEN, return_sequences=True, name='actor_left_encoder')(
        lstm_input_left)
    lstm_out = actor_lstm_model(encoded_tensor)  # reuse lstm_model
    out = actor_dense_model([lstm_out, indicator_input_left])  # reuse dense_model
    left_actor_model = Model(inputs=left_state_input,
                             outputs=out, name='left_actor_model')

    # build lower actor straight
    straight_state_input = Input(shape=(TIME_STEPS, TBD_straight + 3), name='state_straight')
    lstm_input_straight = Lambda(lambda x: x[:, :, :-3], output_shape=(TIME_STEPS, TBD_straight))(straight_state_input)
    indicator_input_straight = Lambda(lambda x: x[:, 0, -3:], output_shape=(3,))(straight_state_input)

    encoded_tensor = layers.LSTM(LSTM_HIDDEN, return_sequences=True, name='actor_straight_encoder')(
        lstm_input_straight)
    lstm_out = actor_lstm_model(encoded_tensor)  # reuse lstm_model
    out = actor_dense_model([lstm_out, indicator_input_straight])  # reuse dense_model
    straight_actor_model = Model(inputs=straight_state_input,
                                 outputs=out, name='straight_actor_model')

    # build lower actor right
    right_state_input = Input(shape=(TIME_STEPS, TBD_right + 3), name='state_right')
    lstm_input_right = Lambda(lambda x: x[:, :, :-3], output_shape=(TIME_STEPS, TBD_right))(right_state_input)
    indicator_input_right = Lambda(lambda x: x[:, 0, -3:], output_shape=(3,))(right_state_input)

    encoded_tensor = layers.LSTM(LSTM_HIDDEN, return_sequences=True, name='actor_right_encoder')(
        lstm_input_right)
    lstm_out = actor_lstm_model(encoded_tensor)  # reuse lstm_model
    out = actor_dense_model([lstm_out, indicator_input_right])  # reuse dense_model
    right_actor_model = Model(inputs=right_state_input,
                              outputs=out, name='right_actor_model')

    # build lower critic shared part--------------------------------------------
    critic_lstm_model = Sequential(name='shared_critic_lstm_model')
    critic_lstm_model.add(layers.LSTM(LSTM_HIDDEN))

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

    encoded_tensor = layers.LSTM(LSTM_HIDDEN, return_sequences=True, name='critic_left_encoder')(
        lstm_input_left)
    lstm_out = critic_lstm_model(encoded_tensor)  # reuse lstm_model
    q_output = critic_dense_model([lstm_out, action_input, indicator_input_left])  # reuse dense_model
    left_critic_model = Model(inputs=[left_state_input, action_input],
                              outputs=q_output, name='left_critic_model')

    # build lower critic straight
    encoded_tensor = layers.LSTM(LSTM_HIDDEN, return_sequences=True, name='critic_straight_encoder')(
        lstm_input_straight)
    lstm_out = critic_lstm_model(encoded_tensor)  # reuse lstm_model
    q_output = critic_dense_model([lstm_out, action_input, indicator_input_straight])  # reuse dense_model
    straight_critic_model = Model(inputs=[straight_state_input, action_input],
                                  outputs=q_output, name='straight_critic_model')

    # build lower critic right
    encoded_tensor = layers.LSTM(LSTM_HIDDEN, return_sequences=True, name='critic_right_encoder')(
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
    return model_dict, left_state_input

model_dict, left_state_input = build_models()
upper_model = model_dict['upper_model']

left_actor_model = model_dict['left_actor_model']
left_critic_model = model_dict['left_critic_model']

straight_actor_model = model_dict['straight_actor_model']
straight_critic_model = model_dict['straight_critic_model']

right_actor_model = model_dict['right_actor_model']
right_critic_model = model_dict['right_critic_model']

print(left_actor_model.input, left_critic_model.input)
print(right_actor_model.input, right_critic_model.input)
print(left_critic_model.input.index(left_state_input))




# tf.keras.utils.plot_model(left_actor_model, 'left_actor_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(straight_actor_model, 'straight_actor_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(right_actor_model, 'right_actor_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(left_critic_model, 'left_critic_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(straight_critic_model, 'straight_critic_model_with_shape_info.png', show_shapes=True)
# tf.keras.utils.plot_model(right_critic_model, 'right_critic_model_with_shape_info.png', show_shapes=True)


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
# memory = SequentialMemory(limit=50000, window_length=1)
# policy = BoltzmannQPolicy()
# dqn = DQNAgent(model=upper_model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#                target_model_update=1e-2, policy=policy, enable_double_dqn=True)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#
# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
#
# # After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#
# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)
