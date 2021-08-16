import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from Classes.MILP_model import MILP_model
from Classes.ClassObjects import A2CMemory, A2CEvaluation

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super().__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear3 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.normalize(state, dim=0)  

        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=0)

        return value, policy_dist

class A2C:
    def __init__(self, env, configuration, torch_seed=None):
        torch.manual_seed(torch_seed)

        self.env = env

        self.num_inputs     = env.state_space.size
        self.num_outputs    = env.action_space.size
        self.num_steps      = env.number_of_steps

        self.hidden_size    = configuration['hidden layer size']
        self.learning_rate  = configuration['learning rate']
        self.GAMMA          = configuration['discount factor']
        self.CRITIC_WEIGHT  = configuration['critic weight']
        self.ACTOR_WEIGHT   = configuration['actor weight']
        self.ENTROPY_WEIGHT = configuration['entropy weight']

    def reset(self):
        self.AC_NN        = ActorCritic(self.num_inputs, self.num_outputs, self.hidden_size)
        self.AC_optimiser = optim.Adam(self.AC_NN.parameters(), lr=self.learning_rate)
        self.memory       = A2CMemory()
        self.episode      = 0

    def run_learning_episode(self):
        log_probs = []
        values    = []
        rewards   = []
        entropies = []

        self.env.reset()
        state = self.env.state_space
        for steps in range(self.num_steps):
            value, policy_dist = self.AC_NN.forward(state)               

            dist   = torch.distributions.Categorical(probs=policy_dist) 
            action = dist.sample()
            new_state, reward, invalid_action, _, _ = self.env.step(action.numpy())

            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()
            
            state = new_state                   

            rewards.append(reward)              
            values.append(value)               
            log_probs.append(log_prob)        
            entropies.append(entropy)
            self.memory.invalid_actions.append(self.episode) if invalid_action else None 
            
            # When the simulation reaches the end of the simulation time-span 
            if steps == self.num_steps-1:
                qval, _ = self.AC_NN.forward(new_state)
                qval    = qval.detach()
                self.memory.all_rewards.append(np.sum(rewards))                 
                print("episode: {}, reward: {}\n".format(self.episode, np.sum(rewards)))
        
        # compute Q-values values
        qvals = np.zeros(len(values))
        for t in reversed(range(len(rewards))):
            qval     = rewards[t] + self.GAMMA * qval
            qvals[t] = qval
  
        # --- update actor critic ---
        log_probs    = torch.stack(log_probs)
        entropies    = torch.stack(entropies)
        values       = torch.stack(values).flatten()
        qvals        = torch.Tensor(qvals)

        advantage    = qvals - values

        # compute losses    
        critic_loss  = advantage.pow(2).mean() * self.CRITIC_WEIGHT      # MSE
        entropy_loss = entropies.sum()         * self.ENTROPY_WEIGHT     # Entropy loss
        actor_loss   = (-log_probs * advantage.detach() - self.ENTROPY_WEIGHT * entropies).mean() * self.ACTOR_WEIGHT  # Actor loss
        total_loss   = critic_loss + actor_loss

        self.memory.append_loss(actor_loss.data.tolist(), 
                                critic_loss.data.tolist(), 
                                entropy_loss.data.tolist(), 
                                total_loss.data.tolist())

        # update parameters
        self.AC_optimiser.zero_grad()
        total_loss.backward()
        self.AC_optimiser.step()

        self.episode += 1
    
    def run_trained_episode(self):
        eval = {}
        rewards = []

        self.env.reset()
        state = self.env.state_space
        for steps in range(self.num_steps):
            _, policy_dist = self.AC_NN.forward(state)               

            dist   = torch.distributions.Categorical(probs=policy_dist) 
            action = dist.sample()
            new_state, reward, invalid_action, new_profit, baseline_profit = self.env.step(action.numpy())

            action_dict = {}
            if action < (self.env.action_space.size - 1):
                action_dict['type']   = self.env.Fleet.types[action//(self.env.max_number_of_actions * 2)]
                action_dict['number'] = self.env.action_space[action]
            elif action == (self.env.action_space.size - 1):
                action_dict         = 'no action'
        
            eval['step '+str(steps+1)] = A2CEvaluation(state, action_dict, reward, new_profit, baseline_profit)
            rewards.append(reward)
            
            state = new_state  

            # When the simulation reaches the end of the simulation time-span 
            if steps == self.num_steps-1:
                qval, _ = self.AC_NN.forward(new_state)
                qval    = qval.detach()                
                print(f'episode: {self.episode}, total rewards: {sum(rewards)}\n')                 

        self.episode += 1

        return eval

    def show_learning_progress(self, interval, num_episodes):
        # --- plot (intermediate) results ---
        if (self.episode % interval == 0) or (self.episode == num_episodes - 1):
            smoothend_rewards = pd.Series.rolling(pd.Series(self.memory.all_rewards), 100).mean()
            smoothend_rewards = [elem for elem in smoothend_rewards]

            plt.figure('rewards', figsize=(12, 7))
            plt.plot(self.memory.all_rewards, 'kx')
            plt.plot(smoothend_rewards, 'r')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.hlines(0, 0, self.episode)
            plt.title(f'Last weighted average value: {round(smoothend_rewards[-1], 5)}')
            plt.pause(1e-12)

            dict_data = {'actor loss':      [self.memory.all_actor_loss, self.ACTOR_WEIGHT],
                         'critic loss':     [self.memory.all_critic_loss, self.CRITIC_WEIGHT],
                         'entropy loss':    [self.memory.all_entropies_loss, self.ENTROPY_WEIGHT], 
                         'total loss':      [self.memory.all_total_loss, '-']}
            
            for label in dict_data.keys():
                smoothend_rewards = pd.Series.rolling(pd.Series(dict_data[label][0]), 50).mean()
                smoothend_rewards = [elem for elem in smoothend_rewards]

                plt.figure(label)
                plt.title(label + ' weight: ' + str(dict_data[label][1]))
                plt.plot(dict_data[label][0], 'kx')
                plt.plot(smoothend_rewards, 'r')
                plt.xlabel('Episode')
                plt.ylabel(label)
                plt.hlines(0, 0, self.episode)
                plt.pause(1e-12)

