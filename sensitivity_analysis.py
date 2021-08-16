import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Classes.MILP_model import MILP_model
from Classes.A2C import A2C

def run_sensitivity_simulations(num_episodes, sensitivity_param, env, A2C_baseline_configuration, pickle_data=False):
    """
    Function to perform several learning simulations; one for each trial as stated in sensitivity_param.
    All rewards are then stored in a pickle.
    """
    sensitivity_dict = {}

    # run baseline problem
    a2c = A2C(env, A2C_baseline_configuration, torch_seed=1)
    a2c.reset()                                             # initialise A2C Neural Network, A2C optimiser, and set episode to 0
    for episode in range(num_episodes):                     # perform baseline learning simulation
        a2c.run_learning_episode()
    sensitivity_dict['baseline'] = a2c.memory.all_rewards   # store all rewards in dictionary

    # run sensitivity problems
    for label in sensitivity_param.keys():
        sensitivity_dict[label] = np.zeros((len(sensitivity_param[label]), num_episodes))   
        
        for i in range(len(sensitivity_param[label])):
            A2C_configuration_copy        = A2C_baseline_configuration.copy()       # adapt baseline problem to single sensitivity problem
            A2C_configuration_copy[label] = sensitivity_param[label][i]             # adapt baseline problem to single sensitivity problem

            a2c = A2C(env, A2C_configuration_copy, torch_seed=1)
            a2c.reset()                                             # initialise A2C Neural Network, A2C optimiser, and set episode to 0
            for episode in range(num_episodes):                     # perform sensitivity learning simulation
                a2c.run_learning_episode()
            sensitivity_dict[label][i, :] = a2c.memory.all_rewards  # store all rewards in dictionary

    # store data in pickle if set to True
    if pickle_data:
        pickle.dump(sensitivity_dict, open('Pickles/sensitivity_all_rewards_N'+str(env_problem_size['network size'])+'2.pkl', "wb"))

def plot_sensitivity_results(num_episodes, A2C_configuration, sensitivity_param, env_problem_size):
    """
    Function to plot the learning performance of the sensitivity analysis
    """
    # load data from pickle, generated in above function
    sensitivity_dict = pickle.load(open("Pickles/sensitivity_all_rewards_N5.pkl", "rb"))

    label_dict = {'learning rate':      '$\u03B1$',
                  'discount factor':    '$\gamma$',
                  'hidden layer size':  '$n_h$',
                  'entropy weight':     '$\epsilon$'}

    exp_mov_aver_span = 50  
    rolling_span      = 1000

    # compute EMA, and smoothend line of the EMA, for the baseline problem
    baseline_rewards              = sensitivity_dict['baseline']
    baseline_EMA_rewards          = pd.DataFrame(baseline_rewards).ewm(span=exp_mov_aver_span).mean().values.flatten()              
    baseline_smoothend_rewards    = pd.DataFrame(baseline_EMA_rewards).rolling(rolling_span, center=True).mean().values.flatten()   

    sensitivity_results = {}

    # compute EMA, and smoothend line of the EMA, for the sensitivity analysis
    for label in sensitivity_dict.keys():
        if label != 'baseline':
            #                            ;Rewards                                                   ; Exponential moving average                                ; smoothend rewards
            sensitivity_results[label] = [np.zeros((sensitivity_dict[label].shape[0], num_episodes)) , np.zeros((sensitivity_dict[label].shape[0], num_episodes)) , np.zeros((sensitivity_dict[label].shape[0], num_episodes))]
            for i in range(sensitivity_results[label][0].shape[0]):
                rewards           = sensitivity_dict[label][i, :]
                EMA_rewards       = pd.DataFrame(rewards).ewm(span=exp_mov_aver_span).mean().values.flatten()           
                smoothend_rewards = pd.DataFrame(EMA_rewards).rolling(rolling_span, center=True).mean().values.flatten()

                sensitivity_results[label][0][i, :] = rewards           
                sensitivity_results[label][1][i, :] = EMA_rewards 
                sensitivity_results[label][2][i, :] = smoothend_rewards 

    # plot results
    for label in sensitivity_results.keys():
        plt.figure(label + ' sensitivity N5', figsize=(12, 7))
        plt.title(label.capitalize() + ' sensitivity analysis, network size N = 5', fontsize=16)
        plt.rcParams.update({'font.size': 14})

        plt.plot(baseline_EMA_rewards, 'black', alpha = 0.2)
        plt.plot(baseline_smoothend_rewards, 'black', alpha = 1, label = 'baseline, '+label_dict[label]+' = '+str(A2C_configuration[label]))

        for i in range(sensitivity_results[label][0].shape[0]):
            p = plt.plot(sensitivity_results[label][1][i, :], alpha = 0.2)
            color = p[0].get_color()
            plt.plot(sensitivity_results[label][2][i, :], color, alpha = 1, label = label_dict[label]+' = '+str(sensitivity_param[label][i]))

        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Exponential moving average of the rewards', fontsize=14)
        plt.plot(np.zeros(num_episodes), 'k', linewidth=1)
        plt.grid()
        plt.legend(fontsize=14)
        plt.xlim(0, num_episodes)
        plt.ylim(-0.065, 0.008)
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":

    # **************************** Environment setup ****************************
    env_problem_size = {'network size':         5,      
                        'aircraft types':       3,
                        'simulation time span': 1,      # in years
                        'number of steps':      5,
                        'number of actions':    4}       # that is, the agent can decide to remove or add a maximum of 'number of actions' aircraft of a single aircraft type each simulation step

    env_configuration = {'lease cost weight':           0.8,
                         'invalid action penalty':      -0.05,
                         'demand scale':                20,
                         'demand stdv scale':           1/30,
                         'average demand growth':       1.01,
                         'initial fleet adjustment':    np.zeros(env_problem_size['aircraft types']).astype('int')}
                        #  'initial fleet adjustment':    np.random.random_integers(-5, 5, size=env_problem_size['aircraft types'])}

    env = MILP_model(env_problem_size, env_configuration, numpy_seed=1)
    # ****************************************************************************

    # **************************** A2C baseline setup ****************************
    A2C_baseline_configuration = {'learning rate':      1e-4,
                                  'discount factor':    0.95,
                                  'hidden layer size':  100,
                                  'critic weight':      0.5,
                                  'actor weight':       1.0,
                                  'entropy weight':     1e-3}
    # ****************************************************************************

    # ************* Learning Parameteres Sensitivity Analysis setup *************
                                               #[[Values to try         ], Zero arrays to store simulation rewards in
    sensitivity_param = {'learning rate':        [4e-5, 8e-5, 2e-4, 6e-4],
                         'discount factor':      [0.8 , 0.85, 0.9 , 1.0 ],
                         'hidden layer size':    [25  , 50  , 75  , 125 ],
                        #  'critic weight':        [0.01, 0.1 , 1.0 , 2.0 ],
                        #  'actor weight':         [0.1 , 0.5 , 1.5 , 2.0 ],
                         'entropy weight':       [0   , 5e-4, 5e-3, 0.01]}
    # ****************************************************************************

    num_episodes = 15_000

    # run_sensitivity_simulations(num_episodes, sensitivity_param, env, A2C_baseline_configuration, pickle_data=False)
    plot_sensitivity_results(num_episodes, A2C_baseline_configuration, sensitivity_param, env_problem_size)
    print()
