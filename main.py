import numpy as np
import matplotlib.pyplot as plt

from Classes.MILP_model import MILP_model
from Classes.A2C import A2C

from sensitivity_analysis import run_sensitivity_simulations, plot_sensitivity_results
from train_agent import run_training_simulations, plot_training_results, compute_average_training_time
from evaluate_trained_model import run_trained_model, plot_actions, plot_profits

def main():
    # **************************** Environment setup ****************************
    env_problem_size = {'network size':             10,     # pickles contain data for values equal to 5 and 10
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

    # *************************** A2C baseline setup *****************************
    A2C_baseline_configuration = {'learning rate':        1e-4,
                                  'discount factor':      0.95,
                                  'hidden layer size':    100,
                                  'critic weight':        0.5,
                                  'actor weight':         1.0,
                                  'entropy weight':       1e-3}
    # ****************************************************************************





    ''' Perform Sensitivity Analysis'''
    num_episodes = 15_000

    # ************* Learning Parameteres Sensitivity Analysis setup *************
    sensitivity_param = {'learning rate':        [4e-5, 8e-5, 2e-4, 6e-4],
                         'discount factor':      [0.8 , 0.85, 0.9 , 1.0 ],
                         'hidden layer size':    [25  , 50  , 75  , 125 ],
                        #  'critic weight':        [0.01, 0.1 , 1.0 , 2.0 ],
                        #  'actor weight':         [0.1 , 0.5 , 1.5 , 2.0 ],
                         'entropy weight':       [0   , 5e-4, 5e-3, 0.01]}
    # ****************************************************************************

    # run_sensitivity_simulations(num_episodes, sensitivity_param, env, A2C_baseline_configuration, pickle_data=False)
    plot_sensitivity_results(num_episodes, A2C_baseline_configuration, sensitivity_param, env_problem_size)





    ''' Perform several Monte Carlo training simulations with best learning parameters'''
    # ******************************** A2C setup *********************************
    A2C_best_configuration = {'learning rate':        2e-4,
                              'discount factor':      0.90,
                              'hidden layer size':    100,
                              'critic weight':        0.5,
                              'actor weight':         1.0,
                              'entropy weight':       5e-4}

    a2c = A2C(env, A2C_best_configuration, torch_seed=1)
    # ****************************************************************************

    num_MC = 10

    # run_training_simulations(num_MC, num_episodes, a2c, show_learning_progress=False, pickle_data=False)
    plot_training_results(num_MC, num_episodes, env_problem_size)
    median_train_time = compute_average_training_time(num_MC, num_episodes, env_problem_size)
    print(f'average train time: {round(median_train_time,2)} sec')




    ''' Perform evaluation of trained model'''
    num_trained_episodes = 50       # choose 50 to plot pickled data; else, first generate new data

    # run_trained_model(num_trained_episodes, num_MC, a2c, env_problem_size, pickle_data=False)
    plot_actions(num_trained_episodes, env_problem_size)
    plot_profits(num_trained_episodes, env_problem_size)
    print()


if __name__ == "__main__":
    main()