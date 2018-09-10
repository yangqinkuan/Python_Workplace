"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v0')   # 定义使用gym库之中的环境
env = env.unwrapped

print(env.action_space)  #查看环境可用的ACTION有多少个
print(env.observation_space) #查看这个环境中可用的state的observation有多少个
print(env.observation_space.high)  # 查看observation 最高取值
print(env.observation_space.low)   # 查看 observation 最低取值

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0

for i_episode in range(100):

    obervation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(obervation)
        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2
        RL.store_transition(obervation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' spsilon: ',round(RL.epsilon, 2))
            break

        obervation = observation_
        total_steps += 1

RL.plot_cost()