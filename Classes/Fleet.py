import pandas as pd

from Classes.ClassObjects import FleetInfo

class Fleet:
    def __init__(self, number_of_aircraft_types):
        if number_of_aircraft_types > 4:
            raise KeyError('number_of_aircraft_types should be less or equal than 4')

        aircraft_data = pd.read_excel("Data/aircraft_data.xlsx", usecols = 'A:E', skiprows = 1, index_col = 0).T
        aircraft_data = aircraft_data.iloc[0:number_of_aircraft_types, :]

        self.types = aircraft_data.index.to_numpy()

        self.info = {}
        for type in self.types:
            info = aircraft_data.loc[type, :]
            self.info[type] = FleetInfo(info['Speed'], info['Seats'], info['Average TAT']/60, info['Maximum range'],
                                        info['Runway required'], info['Weekly lease cost'], info['Fixed operating cost'],
                                        info['Time cost parameter'], info['Fuel cost parameter'])
        
        self.init_config = None     # The initial configuration is added by solving the MILP fleet optimiser model at the environment initialisation
        self.config      = None     # The fleet configuration is added by solving the MILP fleet optimiser model

    def take_action(self, type, number):
        new_number_of_type = self.config[type] + number
        config             = self.config.copy()
        if new_number_of_type >= 0:
            config[type]   = new_number_of_type
            invalid_action = False
        elif new_number_of_type < 0:
            invalid_action = True
            pass

        return config, invalid_action

