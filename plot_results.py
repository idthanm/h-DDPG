import os
import json
import matplotlib.pyplot as plt


def weighted_decay(orig_list, epsilon):
    moving_avg = 0
    processed_list = []
    for value in orig_list:
        moving_avg = value + epsilon * moving_avg
        processed_list.append(moving_avg)
    assert len(processed_list) == len(orig_list)
    return processed_list


path = os.path.dirname(__file__) + os.sep + 'rl' + os.sep + 'log.json'

with open(path, "r") as file:
    fileJson = json.load(file)
    loss = fileJson["loss"]
    mae = fileJson["mae"]
    mean_q = fileJson["mean_q"]
    left_loss = fileJson["left_loss"]
    left_mae = fileJson["left_mae"]
    left_mean_q = fileJson["left_mean_q"]
    straight_loss = fileJson["straight_loss"]
    straight_mae = fileJson["straight_mae"]
    straight_mean_q = fileJson["straight_mean_q"]
    right_loss = fileJson["right_loss"]
    right_mae = fileJson["right_mae"]
    right_mean_q = fileJson["right_mean_q"]
    episode_reward = fileJson["episode_reward"]
    nb_episode_steps = fileJson["nb_episode_steps"]
    nb_steps = fileJson["nb_steps"]
    memory_len = fileJson["memory_len"]
    episode = fileJson["episode"]
    duration = fileJson["duration"]

# ——————————————————
plt.figure('losses and metrics')
plt.subplot(431)
plt.plot(episode, loss)
plt.title('upper_loss')

plt.subplot(432)
plt.plot(episode, mae)
plt.title('upper_mae')

plt.subplot(433)
plt.plot(episode, mean_q)
plt.title('upper_mean_q')
# ————————————————
plt.subplot(434)
plt.plot(episode, left_loss)
plt.title('lower_left_loss')

plt.subplot(435)
plt.plot(episode, left_mae)
plt.title('lower_left_mae')

plt.subplot(436)
plt.plot(episode, left_mean_q)
plt.title('lower_left_mean_q')
# ————————————————
plt.subplot(437)
plt.plot(episode, straight_loss)
plt.title('lower_straight_loss')

plt.subplot(438)
plt.plot(episode, straight_mae)
plt.title('lower_straight_mae')

plt.subplot(439)
plt.plot(episode, straight_mean_q)
plt.title('lower_straight_mean_q')
# ————————————————
plt.subplot(4, 3, 10)
plt.plot(episode, right_loss)
plt.title('lower_right_loss')

plt.subplot(4, 3, 11)
plt.plot(episode, right_mae)
plt.title('lower_right_mae')

plt.subplot(4, 3, 12)
plt.plot(episode, right_mean_q)
plt.title('lower_right_mean_q')
plt.subplots_adjust(wspace=0.2, hspace=0.8)

# ————————————————————
plt.figure('misc')
plt.subplot(221)
plt.plot(episode, episode_reward)
plt.title('episode_reward')

plt.subplot(222)
EPSILON = 0.99
plt.plot(episode, weighted_decay(episode_reward, EPSILON))
plt.title('{}weighted_decay_reward'.format(str(EPSILON)))

plt.subplot(223)
plt.plot(episode, nb_steps)
plt.title('total steps')

plt.subplot(224)
plt.plot(episode, nb_episode_steps)
plt.title('episode steps')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()



