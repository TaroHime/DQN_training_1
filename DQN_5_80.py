# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:37:39 2022

@author: prosu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1)



from torch import nn
from torch import optim
import torch
import numpy as np
torch.manual_seed(1)

import time

# 処理前の時刻
t1 = time.time() 

maze_size=80

class Environment:
    
    def __init__(self, size=maze_size, lucky=[]):
        
        self.size = size
        self.lucky = lucky
        self.goal = (size-40, size-40)
      #  self.goal = (2, 3)3
        
        self.states = [(x, y) for x in range(size) for y in range(size)]
            
    def next_state(self, s, a):
        
        s_next = (s[0] + a[0], s[1] + a[1])
        
        if s == self.goal:
            return s
        
        if s_next not in self.states:
            return s
        
      #  if s_next in self.lucky:
      #      if np.random.random() < 0.8:
      #         return self.goal
      #      else:
      #          return s_next
        
        return s_next
    
    def reward(self, s, s_next):
        
        if s == self.goal:
            return -1
        
        if s_next == self.goal:
            return 0
        
        return -1

class Agent():
    
    def __init__(self, environment):
        
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.environment = environment
                
    def action(self, s, a, prob=False):

        s_next = self.environment.next_state(s, a)
        r = self.environment.reward(s, s_next)
      #  print(r)
        return r, s_next






    
 
class NN():    
    def __init__(self, agent):
        self.model = self.model()
        self.criterion = nn.MSELoss()
        self.actions = agent.actions
        
    def model(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(4,  16))    
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(16, 32))     
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc3', nn.Linear(32, 16))     
        model.add_module('relu3', nn.ReLU())   
                               
        model.add_module('fc4', nn.Linear(16, 1))
        self.optimizer = optim.Adam(model.parameters())
        return model
    
   # def train_model(self, sa, labels, num_train=1000):
    def train_model(self, sa, labels, num_train=1000):    
        for _ in range(num_train):
            qvalue = self.model(torch.tensor(sa).float())#予測値
            loss = self.criterion(qvalue, torch.tensor(labels).float())#誤差計算
            #予測値　　正解の順
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def q_max(self, state):  #amax qmax retuen
        sa = []
        for action in self.actions:
            sa.append(state+action)
        q = self.model(torch.tensor([np.array(sa)]).float()).detach()
        #print("q:",q)
        a_max = np.argmax(q)
        #print('a_max:',a_max) #tensor(0) or tensor(1) or tensor(2) or tensor(3)
        #print('q:',q)#[-0.2746],[-0.2917],[-0.3497],[-0.3447]
        #print('q[0,a_max,0] :',q[0,a_max,0] )
        return self.actions[a_max], q[0,a_max,0]    #左記なぜか　q max
    
    
    
def get_episode(agent, nn_model, epsilon=0.1):
    
    s = agent.environment.states[np.random.randint(agent.environment.size**2-1)]
    #print(agent.environment.size**2-1)   224    =15+15-1
    #  print(s)
   #print(agent.environment.states[1])  ランダムに位置を選択　ex (0,0)  
    episode = []
    step=0
    while True:
        
        if np.random.random() < epsilon:
          #  a = agent.actions[np.random.randint(2,4)]
            a = agent.actions[np.random.randint(0,4)]
        else:
            a, _ = nn_model.q_max(s)
            
        r, s_next = agent.action(s, a)
        
        episode.append((s, a, r, s_next))   
       # print(a)  →　(-1,0)  (1,0)  (0,-1) (0,1) 
        if s_next == agent.environment.goal:
            break
      #  if step>=100:
        if step>=400:    
            #print('100')
            
            break
        
        s = s_next
        
        step+=1
        
    return episode


def train(agent, nn_model, epsilon=0.1, num=100, num_train=1000):
#def train(agent, nn_model, epsilon=0.1, num=100, num_train=3000):    
    print(num)
    for c in range(num):
        print(f'num : {c+1} ')
        #計算用のサンプルを取得する処理　　100個暫定
        examples = []
       # for _ in range(200):
        for _ in range(100):    
            episode = get_episode(agent, nn_model, epsilon)
           # print(episode[0])  #episode=s,a,r,next_s
            #print(episode)  #ゴールするまでの全ﾃﾞｰﾀ
            examples += episode
            #print(examples)  # ? (1,1),(0,1),0,(3,3)   s,a, reward, s_next
            
        np.random.shuffle(examples)
        print("len",len(examples),sep='-')
        sa = []
        labels = []
        for s, a, r, s_next in examples:
            sa.append(s+a)
            _, q_next = nn_model.q_max(s_next)
            #print('sa',sa,sep='-') #(2,5,0,-1)
            #print('_',_,sep='-') #
            #print('q_next',q_next,sep='-') #
            labels.append([r + q_next.detach()])
            #print('Labels',labels,sep='-')  #tensor(-1.2995)
            #   print(sa)   (0,1, 0,-1), (3, 1, 0, -1)位置とnext action
          #  print(labels[0])
          #  print(sa[0])
        #NN学習
        nn_model.train_model(sa, labels, num_train)
    
 
    
    
    
def show_maze(environment):
    size = environment.size
    fig = plt.figure(figsize=(3,3))

    plt.plot([-0.5, -0.5], [-0.5, size-0.5], color='k')
    plt.plot([-0.5, size-0.5], [size-0.5, size-0.5], color='k')
    plt.plot([size-0.5, -0.5], [-0.5, -0.5], color='k')
    plt.plot([size-0.5, size-0.5], [size-0.5, -0.5], color='k')
    
    for i in range(size):
        for j in range(size):
            plt.text(i, j, "{}".format(i+size*j), size=5, ha="center", va="center")
            if (i,j) in environment.lucky:
                x = np.array([i-0.5,i-0.5,i+0.5,i+0.5])
                y = np.array([j-0.5,j+0.5,j+0.5,j-0.5])
                plt.fill(x,y, color="lightgreen")

    plt.axis("off")    
    plt.show()

def show_values(agent, nn_model):

    fig = plt.figure(figsize=(3,3))
    result = np.zeros([agent.environment.size, agent.environment.size])
    for (x, y) in agent.environment.states:
        a_max, q_max =  nn_model.q_max((x, y))
        result[y][x]  = q_max
    #   print(x,y)        
    #    print(a_max[0])
    #    print(a_max[1])
      
        
    sns.heatmap(result, square=True, cbar=False, annot=True, fmt='3.2f', cmap='autumn_r').invert_yaxis()
    plt.axis("off")    
    plt.show()
    
    
    
    
    
def show_policy(agent, nn_model):
    size = agent.environment.size
    fig = plt.figure(figsize=(3,3))

    plt.plot([-0.5, -0.5], [-0.5, size-0.5], color='k')
    plt.plot([-0.5, size-0.5], [size-0.5, size-0.5], color='k')
    plt.plot([size-0.5, -0.5], [-0.5, -0.5], color='k')
    plt.plot([size-0.5, size-0.5], [size-0.5, -0.5], color='k')

    for i in range(size):
        for j in range(size):
            if (i,j) in agent.environment.lucky:
                x = np.array([i-0.5,i-0.5,i+0.5,i+0.5])
                y = np.array([j-0.5,j+0.5,j+0.5,j-0.5])
                plt.fill(x,y, color="lightgreen")

    rotation = {(-1, 0): 180, (0, 1): 90, (1, 0): 0, (0, -1): 270}
    for s in agent.environment.states:
        if s == agent.environment.goal:
            direction=None
        else:
            a_max, q_max =  nn_model.q_max(s)
            direction = rotation[a_max]
        
        if direction != None:
            bbox_props = dict(boxstyle='rarrow')
            plt.text(s[0], s[1], '     ', bbox=bbox_props, size=2,
                     ha='center', va='center', rotation=direction)
                        
    plt.axis("off")    
    plt.show()


    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#A=torch.randn(3,5)
#A.to(device)





#env1 = Environment(size=20, lucky=[(1,2), (1,3)]) #2min
env1 = Environment(size=maze_size, lucky=[(1,2), (1,3)]) #2min


agent1 = Agent(env1)
#model=NN(agent1).to(device)

model1=NN(agent1)


#model1.to(device)

#show_maze(env1)    

#train(agent1, model1,epsilon=0.5,num=100)#num=100  size=30 収束しない
train(agent1, model1,epsilon=0.5,num=500)#num=      size=30 収束しない
show_policy(agent1, model1)
#show_values(agent1, model1)



torch.save(model1,'DQN_5_80.ckpt')  


# 処理後の時刻
t2 = time.time() 
# 経過時間を表示
elapsed_time = (t2-t1)/60
print(f"経過時間(min)：{elapsed_time}")






