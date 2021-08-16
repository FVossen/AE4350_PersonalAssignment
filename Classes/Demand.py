import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Demand:
    def __init__(self, network_size, simulation_time_span, demand_scale, demand_stdv_scale, average_network_growth):
        if network_size > 20:
            raise KeyError('network_size should be less or equal than 20')

        self.network_size           = network_size
        self.simulation_time_span   = simulation_time_span
        self.demand_scale           = demand_scale
        self.stdv_scale             = demand_stdv_scale
        self.average_network_growth = average_network_growth   
        self.reset()

    def reset(self):
        self.values = pd.read_excel("Data/demand_data.xlsx", index_col = 0, usecols = 'A:U', nrows = 21)    # with unit: demand per week
        self.values = self.values.iloc[0:self.network_size, 0:self.network_size]
        self.values *= 52 * self.simulation_time_span * self.demand_scale           # convert weekly demand to yearly demand; and add a scaling factor
        self.values = self.values.astype('int')
        self.old_values = self.values - (self.random() - self.values)               # perform one random step backwards

    def random(self):
        stdv = self.values * self.stdv_scale
        mean = self.values * self.average_network_growth
        random_demand = np.random.normal(mean, stdv, size=self.values.shape)        # Sample random demand from normal distribution with: mean = old demand value; std = old demand value/6
        random_demand = random_demand.clip(min=0)                                   # Set all negative values equal to zero
        random_demand = np.tril(random_demand) + np.triu(random_demand.T, 1)        # Mirror elements to make demand matrix symmetric
        np.fill_diagonal(random_demand, 0)   
        ICAO          = self.values.index.to_numpy()
        random_demand = pd.DataFrame(random_demand, columns=ICAO, index=ICAO).astype('int') # Turn numpy array into pandas DataFrame

        return random_demand
