# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:48:54 2022

@author: nrrdk
"""

import gym
# 環境の生成 

#env = gym.make('CartPole-v1')
env = gym.make("CartPole-v1", render_mode='human')
env.reset()
#im = env.render(width=200, height=200)
#im2 = env.render()
# 環境の初期か
observation = env.reset()
#env.render('human')

for t in range(100):
    # 現在の状況を表示させる
    env.render()
    # サンプルの行動をさせる　返り値は左から台車および棒の状態、得られた報酬、ゲーム終了フラグ、詳細情報
    observation, reward, done, info,d = env.step(env.action_space.sample())
    if done:
        print("Finished after {} timesteps".format(t+1))
        break
# 環境を閉じる

env.close()