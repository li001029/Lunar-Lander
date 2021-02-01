import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import time
import numpy as np

class replaymemory:
    def __init__(self):
 
        self.statememory=deque(maxlen=1000000)
        self.rewardmemory=deque(maxlen=1000000)
        self.nextstatememory=deque(maxlen=1000000)
        self.actionmemory=deque(maxlen=1000000)
        self.donememory=deque(maxlen=1000000)
        self.size=0
    def remember(self,state,action,reward,nextstate,done):
      
        self.statememory.append(state)
        self.rewardmemory.append(reward)
        self.nextstatememory.append(nextstate)
        self.actionmemory.append(action)
        self.donememory.append(done)
        self.size+=1
        
class Agent:
    def __init__(self,epsilon,gamma,batchsize,
                 epsilonmin,epsilondecay,learningrate):
        
        self.action_space = 4
        self.state_space = 8
        #initial epsilon value set to be 1
        self.epsilon = epsilon
        #reward discount
        self.gamma = gamma
        self.batchsize = batchsize
        #minimum epsilon used to compare with current epsilon and
        #decide what action decision to make
        self.epsilonmin = epsilonmin
        self.lr = learningrate
        #current epsilon times epsilon decay rate is next epsilon
        self.epsilondecay = epsilondecay
        self.network = self.neuralnetwork()

    def neuralnetwork(self):
        network = Sequential()
        #input_dim=state.space is the dimension of our input, and our
        #first hidden layer would have 500 units
        network.add(Dense(500, input_dim=self.state_space,
                                          activation=relu))
        #second hidden layer would have 250 units
        network.add(Dense(250, activation=relu))
        #number of units in output layer should be the same as action_space
        #dimension, that is, 4
        network.add(Dense(self.action_space,
                                          activation=linear))
        network.compile(loss='mse', optimizer=adam(lr=self.lr))
        return network



        
#Exploration-exploitation trade-off
    def chooseaction(self, statespace):
        k=random.uniform(0,1)
        #if probability is greater than epsilon then use predicted action 
        if k > self.epsilon:
            actionquality = self.network.predict(statespace)
            bestaction=np.argmax(actionquality[0])
            return bestaction
        #otherwise random choose action
        if k<=self.epsilon:
            return random.randrange(self.action_space)
        

    def replay(self,replaymemory):
        if replaymemory.size < self.batchsize:
            return
        size=[i for i in range(replaymemory.size)]
        ind2=random.sample(size,self.batchsize)
        states=[]
        actions=[]
        rewards=[]
        next_states=[]
        dones=[]
        for u in ind2:
            states+=[replaymemory.statememory[u]]
            actions+=[replaymemory.actionmemory[u]]
            rewards+=[replaymemory.rewardmemory[u]]
            next_states+=[replaymemory.nextstatememory[u]]
            dones+=[replaymemory.donememory[u]]
        states=np.array(states)
        actions=np.array(actions)
        rewards=np.array(rewards)
        next_states=np.array(next_states)
        dones=np.array(dones)
        
        
        states =np.squeeze(states)
       
        next_states = np.squeeze(next_states)
        t0=time.time()
        prediction=self.network.predict_on_batch(next_states)
        Q=np.amax(prediction, axis=1)
       
        targets = rewards + self.gamma*(Q)*(1-dones)
        
        t1=time.time()
        targets_full = self.network.predict_on_batch(states)
        t2=time.time()

        #print('part1 is',t1-t0)
        #print('part2 is',t2-t1)
        ind = np.array([i for i in range(self.batchsize)])
       
        
        targets_full[[ind], [actions]] = targets
       
        self.network.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilonmin:
            self.epsilon *= self.epsilondecay




if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    loss = []
    buffermemory=replaymemory()
    
    agent = Agent(epsilon=1,gamma=0.99,batchsize=64,epsilonmin=0.01,
                  epsilondecay=0.993,learningrate=0.001)
    for episode in range(0,1000):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        for i in range(3000):       
            action = agent.chooseaction(state)
            env.render()
            next_state, reward, done, info = env.step(action)
    
            next_state = np.reshape(next_state, (1, 8))
            buffermemory.remember(state, action, reward, next_state, done)
            agent.replay(buffermemory)
            score += reward
            state = next_state
            
            if done:
                print('episode:',episode,'/',1000, 'score:',score)
                break
        loss+=[score]

        # Average score of last 50 episode
        average = np.average(loss[-50:])
        if average > 200:
            env.close()
            break
        else:
            print("Average over last 50 episode:",average)
    

    #plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    #plt.show()
