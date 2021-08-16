import pickle
from numpy.core.shape_base import stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from Classes.MILP_model import MILP_model
from Classes.A2C import A2C

def run_training_simulations(num_MC, num_episodes, a2c, show_learning_progress=False, pickle_data=False):
    """
    Function to perform several learning simulations; one for each MC simulation.
    All rewards, actor loss, critic loss, entropy loss, total loss, and train time, are then stored in a pickle.
    The A2C parameters for the trained model, that is the NN weights and biases, are also stored in a pickle.
    """

    MC_memories = {'rewards':       np.zeros((num_MC, num_episodes)),
                   'actor loss':    np.zeros((num_MC, num_episodes)),
                   'critic loss':   np.zeros((num_MC, num_episodes)),
                   'entropy loss':  np.zeros((num_MC, num_episodes)),
                   'total loss':    np.zeros((num_MC, num_episodes)),
                   'train time':    np.zeros((num_MC, num_episodes))}

    MC_state_dicts = []

    # run MC simulations
    for MC in range(num_MC):
        a2c.reset()     # initialise A2C Neural Network, A2C optimiser, and set episode to 0
        for episode in range(num_episodes):     # perform baseline learning simulation

            start_time = time.time()    
            a2c.run_learning_episode()
            end_time = time.time()
            
            a2c.show_learning_progress(100, num_episodes) if show_learning_progress else None

            MC_memories['train time'][MC, episode] = end_time - start_time

        MC_memories['rewards'][MC, :]       = a2c.memory.all_rewards
        MC_memories['actor loss'][MC, :]    = a2c.memory.all_actor_loss
        MC_memories['critic loss'][MC, :]   = a2c.memory.all_critic_loss
        MC_memories['entropy loss'][MC, :]  = a2c.memory.all_entropies_loss
        MC_memories['total loss'][MC, :]    = a2c.memory.all_total_loss

        MC_state_dicts.append(a2c.AC_NN.state_dict())       # Store A2C trained model parameters

        # store data in pickle if set to True
        if pickle_data:
            pickle.dump(MC_memories     , open('Pickles/MC_memories_N'+str(env_problem_size['network size'])+'2.pkl', "wb"))
            pickle.dump(MC_state_dicts  , open('Pickles/MC_state_dicts_N'+str(env_problem_size['network size'])+'2.pkl', "wb"))

def plot_training_results(num_MC, num_episodes, env_problem_size):
    """
    Function to plot the learning performance of the MC simulations
    """
    # load data from pickle, generated in above function
    MC_memories    = pickle.load(open("Pickles/MC_memories_N"+str(env_problem_size['network size'])+".pkl", "rb"))

    exp_mov_aver_span = 50
    rolling_span      = 1000

    dict_data = {'rewards':             np.zeros((num_MC, num_episodes)),
                 'EMA_rewards':         np.zeros((num_MC, num_episodes)),
                 'smoothend_rewards':   np.zeros((num_MC, num_episodes))}

    # compute EMA, and smoothend line of the EMA, for the MC simulations
    for MC in range(num_MC):
        rewards           = np.array(MC_memories['rewards'][MC, :])
        EMA_rewards       = pd.DataFrame(rewards).ewm(span=exp_mov_aver_span).mean().values.flatten()
        smoothend_rewards = pd.DataFrame(EMA_rewards).rolling(rolling_span, center=True).mean().values.flatten()

        dict_data['rewards'][MC, :]           = rewards
        dict_data['EMA_rewards'][MC, :]       = EMA_rewards
        dict_data['smoothend_rewards'][MC, :] = smoothend_rewards

    #                                    ;mean values               ; confidence interval values ; min values                ; max values
    dict_plot = {'rewards':              [np.zeros(num_episodes)    , np.zeros(num_episodes)     , np.zeros(num_episodes)    , np.zeros(num_episodes)],
                 'EMA_rewards':          [np.zeros(num_episodes)    , np.zeros(num_episodes)     , np.zeros(num_episodes)    , np.zeros(num_episodes)],
                 'smoothend_rewards':    [np.zeros(num_episodes)    , np.zeros(num_episodes)     , np.zeros(num_episodes)    , np.zeros(num_episodes)]}

    # compute mean values, confidence interval, min values, and max values of the MC simulations
    for t in range(num_episodes):
        for label in dict_data.keys():
            mean_value          = dict_data[label][:, t].mean()
            confidence_interval = 1.96 * dict_data[label][:, t].std() / np.sqrt(dict_data[label][:, t].size)
            min_value           = dict_data[label][:, t].min()
            max_value           = dict_data[label][:, t].max()

            dict_plot[label][0][t] = mean_value
            dict_plot[label][1][t] = confidence_interval
            dict_plot[label][2][t] = min_value
            dict_plot[label][3][t] = max_value

    # plot results
    plt.figure('rewards N'+str(env_problem_size['network size']), figsize=(12, 7))
    plt.rcParams.update({'font.size': 14})
    plt.title('Learning performance results, network size N = '+str(env_problem_size['network size']), fontsize=16)

    # plt.plot(dict_plot['rewards'][0], 'blue')
    # plt.fill_between(range(num_episodes), (dict_plot['rewards'][2]), (dict_plot['rewards'][3]), color='b', alpha=.1)

    plt.plot(dict_plot['EMA_rewards'][0], 'black', alpha = 0.8, label='mean EMA')
    plt.fill_between(range(num_episodes), (dict_plot['EMA_rewards'][2]), (dict_plot['EMA_rewards'][3]), color='black', alpha=.2, label='min and max EMA')
    plt.fill_between(range(num_episodes), (dict_plot['EMA_rewards'][0] - dict_plot['EMA_rewards'][1]), (dict_plot['EMA_rewards'][0] + dict_plot['EMA_rewards'][1]), color='red', alpha=.4, label='95% confidence interval')

    # plt.plot(dict_plot['smoothend_rewards'][0], 'red', label='mean smoothend EMA')
    # plt.fill_between(range(num_episodes), (dict_plot['smoothend_rewards'][0] - dict_plot['smoothend_rewards'][1]), (dict_plot['smoothend_rewards'][0] + dict_plot['smoothend_rewards'][1]), color='red', alpha=.3, label='95% confidence interval')
    # plt.fill_between(range(num_episodes), (dict_plot['smoothend_rewards'][2]), (dict_plot['smoothend_rewards'][3]), color='b', alpha=.1)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Exponential Moving Average of the rewards', fontsize=14)
    plt.plot(np.zeros(num_episodes), 'k', linewidth=1)
    plt.grid()
    plt.xlim(0, num_episodes)
    plt.ylim(-0.065, 0.008) if env_problem_size['network size'] == 5 else plt.ylim(-0.018, 0.005)
    plt.legend(fontsize=14, loc=4)
    plt.tight_layout()

    plt.show()

def compute_average_training_time(num_MC, num_episodes, env_problem_size):
    # load data from pickle, generated in above function
    MC_memories    = pickle.load(open("Pickles/MC_memories_N"+str(env_problem_size['network size'])+".pkl", "rb"))

    train_time = MC_memories['train time']
    median_train_time = np.median(train_time.flatten())
    return median_train_time


if __name__ == "__main__":

    # **************************** Environment setup ****************************
    env_problem_size = {'network size':             10,     # pickles contain data for values equal to 5 and 10    
                        'aircraft types':           3,
                        'simulation time span':     1,      # in years
                        'number of steps':          5,      # in years
                        'number of actions':        4}      #the agent can decide to remove or add a maximum of 'number of actions' aircraft of a single aircraft type each simulation step
    
    env_configuration = {'lease cost weight':           0.8,        # weight to change the aircraft cost
                         'invalid action penalty':      -0.05,      # penalty added to reward when agent takes action that results in a negative number of aircraft in fleet
                         'demand scale':                20,         # weight to change the overal network demand
                         'demand stdv scale':           1/30,       # weight to change the standard deviation from which the stochastic demand is sampled
                         'average demand growth':       1.01,       # yearly demand increase over the entire network
                         'initial fleet adjustment':    np.zeros(env_problem_size['aircraft types']).astype('int')}                 # modify the initial (optimised) fleet, to start from a less optimal situation
                        #  'initial fleet adjustment':    np.random.random_integers(-5, 5, size=env_problem_size['aircraft types'])}  # modify the initial (optimised) fleet, to start from a less optimal situation
    
    env = MILP_model(env_problem_size, env_configuration, numpy_seed=1)
    # ****************************************************************************

    # ******************************** A2C setup *********************************
    A2C_best_configuration = {'learning rate':        2e-4,
                              'discount factor':      0.90,
                              'hidden layer size':    100,
                              'critic weight':        0.5,
                              'actor weight':         1.0,
                              'entropy weight':       5e-4}

    a2c = A2C(env, A2C_best_configuration, torch_seed=1)
    # ****************************************************************************

    num_MC          = 10
    num_episodes    = 15_000

    # run_training_simulations(num_MC, num_episodes, a2c, show_learning_progress=False, pickle_data=False)
    # plot_training_results(num_MC, num_episodes, env_problem_size)
    median_train_time = compute_average_training_time(num_MC, num_episodes, env_problem_size)
    print(f'average train time: {round(median_train_time,2)} sec')
    print()
