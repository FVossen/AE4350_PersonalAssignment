import pickle
import numpy as np
from numpy.core.fromnumeric import mean
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from Classes.MILP_model import MILP_model
from Classes.A2C import A2C

def run_trained_model(num_trained_episodes, num_MC, a2c, env_problem_size, pickle_data=False):
    """
    Function to perform several evaluation simulations on the trained agent.
    Functions stores the actions that the agent has taken and the additional profit that has resulted from that action.
    """

    # Load trained model parameters
    MC_state_dicts  = pickle.load(open("Pickles/MC_state_dicts_N"+str(env_problem_size['network size'])+".pkl", "rb"))
    num_actions     = env_problem_size['number of actions']

    all_data         = np.zeros((num_MC, env_problem_size['aircraft types'], num_actions * 2))
    all_no_action    = np.zeros(num_MC)
    all_agent_profit = np.zeros((num_MC, num_trained_episodes))

    for MC in range(num_MC):
        a2c.reset()     # initialise A2C Neural Network, A2C optimiser, and set episode to 0
        a2c.AC_NN.load_state_dict(MC_state_dicts[MC])   # load trained model parameters to the agent
        a2c.AC_NN.eval()

        episode_dict = {}

        with torch.no_grad():   # no gradient calculation performed; evaluation mode
            for episode in range(num_trained_episodes):     # perform evaluation simulation
                eval = a2c.run_trained_episode()            # returns evaluation parameters
                episode_dict['episode '+str(episode)] = eval

        # store and preprocess actions taken by agent
        type1 = np.zeros((1, num_actions * 2))
        type2 = np.zeros((1, num_actions * 2))
        type3 = np.zeros((1, num_actions * 2))
        no_action = np.zeros(1)

        episode_num = 0
        for episode in episode_dict.keys():
            agent_profit = []
            for step in episode_dict[episode].keys():
                action = episode_dict[episode][step].action
                if action == 'no action':
                    no_action[0] += 1
                else:
                    ind = int(action['number']+num_actions) if action['number'] < 0 else int(action['number']+num_actions-1)
                    if action['type'] == 'type1':
                        type1[0][ind] += 1
                    elif action['type'] == 'type2':
                        type2[0][ind] += 1
                    elif action['type'] == 'type3':
                        type3[0][ind] += 1
                
                # compute additional profit from action taken by agent
                baseline_profit = episode_dict[episode][step].baseline_profit
                new_profit      = episode_dict[episode][step].new_profit
                agent_profit.append(new_profit - baseline_profit)

            all_agent_profit[MC, episode_num] = sum(agent_profit)
            episode_num += 1

        all_data[MC]        = np.concatenate((type1, type2, type3), axis=0)
        all_no_action[MC]   = no_action

    data_dict = {'all_data':            all_data,
                 'all_no_action':       all_no_action,
                 'all_agent_profit':    all_agent_profit}

    # store data in pickle if set to True
    if pickle_data:
        pickle.dump(data_dict, open('Pickles/agent_evaluation_data_dict_'+str(num_trained_episodes)+'_N'+str(env_problem_size['network size'])+'.pkl', "wb"))

def plot_actions(num_trained_episodes, env_problem_size):
    """
    Function to plot the actions taken by the agent from the evaluation simulations
    """
    # load data from pickle, generated in above function
    data_dict           = pickle.load(open("Pickles/agent_evaluation_data_dict_"+str(num_trained_episodes)+'_N'+str(env_problem_size['network size'])+".pkl", "rb"))
    all_data            = data_dict['all_data']
    all_no_action       = data_dict['all_no_action']
    all_agent_profit    = data_dict['all_agent_profit']

    x_labels    = ['-4', '-3', '-2', '-1', '+1', '+2', '+3', '+4']
    labels      = ['type 1', 'type 2', 'type 3']

    # compute mean action taken
    mean_data       = np.mean(all_data, axis=0)
    mean_no_action  = np.mean(all_no_action)

    x     = np.arange(len(x_labels)) * 2
    width = 1.5/len(labels)

    # plot results
    plt.figure('average action plot N'+str(env_problem_size['network size']), figsize=(12, 5))
    plt.rcParams.update({'font.size': 14})
    for i in range(len(labels)):
        plt.bar(x + width*(i - 1), mean_data[i], width = width, label=labels[i])
    plt.bar(x[-1]+(x[-1]-x[-2]), mean_no_action, width = width)

    # Text on the top of each bar 
    vspacing = 2.0 if env_problem_size['network size'] == 5 else 1.0
    hspacing = 0
    for i in range(len(x_labels)):
        for j in range(len(labels)):
            val = str(int(mean_data[j, i])) if mean_data[j, i] < 0.1 else str(mean_data[j, i])
            plt.text(x=x[i] + width*(j - 1) + hspacing, y=mean_data[j, i] + vspacing, s=val, ha='center')
    plt.text(x=x[-1]+(x[-1]-x[-2]) + hspacing, y=mean_no_action + vspacing, s=str(mean_no_action), ha='center')

    plt.title('Average actions taken by agent, network size N = '+str(env_problem_size['network size']), fontsize=16)
    x = np.append(x, x[-1]+(x[-1]-x[-2]))
    x_labels.append('no action')
    plt.xticks(x, labels=x_labels, fontsize=14)
    plt.ylabel('Number of actions', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_profits(num_trained_episodes, env_problem_size):
    """
    Function to plot the profit obtained by the action taken by the agent
    """
    # load data from pickle, generated in above function
    data_dict        = pickle.load(open("Pickles/agent_evaluation_data_dict_"+str(num_trained_episodes)+'_N'+str(env_problem_size['network size'])+".pkl", "rb"))
    all_agent_profit = data_dict['all_agent_profit']

    x_labels = []
    for i in range(len(all_agent_profit)):
        x_labels.append('MC '+str(i+1))
    df = pd.DataFrame(all_agent_profit.T, columns=x_labels)

    # plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [7, 1]})

    plt.rcParams.update({'font.size': 14})
    sns.boxplot(data=df/1e6, ax=ax1)
    # sns.swarmplot(data=df, color=".25", alpha=0.4)
    ax1.grid()
    ax1.set_title('Agent profit evalution boxplot, network size N = '+str(env_problem_size['network size']), fontsize=16)
    ax1.set_ylabel('Agent profit, in millions', fontsize=14)
    ax1.set_ylim(-3, 2) if env_problem_size['network size'] == 5 else None

    df_all = pd.DataFrame(all_agent_profit.flatten(), columns=['all MC'])

    sns.boxplot(data=df_all/1e6, width=0.6, ax=ax2)
    ax2.grid()
    ax2.set_title(' ', fontsize=16)
    # ax2.set_ylabel('Agent profit, in millions', fontsize=14)
    ax2.set_ylim(-3, 2) if env_problem_size['network size'] == 5 else None

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # **************************** Environment setup ****************************
    env_problem_size = {'network size':             5,     # pickles contain data for values equal to 5 and 10
                        'aircraft types':           3,
                        'simulation time span':     1,      # in years
                        'number of steps':          5,      # in years
                        'number of actions':        4}      # the agent can decide to remove or add a maximum of 'number of actions' aircraft of a single aircraft type each simulation step
    
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

    num_episodes         = 15_000
    num_MC               = 10
    num_trained_episodes = 50

    # run_trained_model(num_trained_episodes, num_MC, a2c, env_problem_size, pickle_data=False)
    plot_actions(num_trained_episodes, env_problem_size)
    plot_profits(num_trained_episodes, env_problem_size)
    print()
