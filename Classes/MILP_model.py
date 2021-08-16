import numpy as np
from gurobipy import Model, GRB, quicksum

from Classes.Network import Network
from Classes.Fleet import Fleet
from Classes.Demand import Demand

class MILP_model:
    def __init__(self, problem_size, configuration, numpy_seed=None):
        if problem_size['aircraft types'] > 4:
            raise KeyError('number_of_aircraft_types should be less or equal than 4')
        if problem_size['network size'] > 20:
            raise KeyError('network_size should be less or equal than 20')

        np.random.seed(numpy_seed)

        # Define problem size
        self.network_size             = problem_size['network size']
        self.number_of_aircraft_types = problem_size['aircraft types']
        self.simulation_time_span     = problem_size['simulation time span']            # in years
        self.number_of_steps          = problem_size['number of steps']
        self.max_number_of_actions    = problem_size['number of actions']

        # Environment design parameters:
        self.lease_cost_weight              = configuration['lease cost weight']        # lease_cost_weight
        self.fleet_penalty                  = configuration['invalid action penalty']   # fleet_penalty
        self.demand_scale                   = configuration['demand scale']             # demand_scale
        self.demand_stdv_scale              = configuration['demand stdv scale']        # define standard deviation as scale to demand values 
        self.demand_average_network_growth  = configuration['average demand growth']    # average growth rate on entire network per time span of simulation
        self.fleet_randomiser               = configuration['initial fleet adjustment']

        # Define model constants
        ACTIVE_TIME_hrs_per_day         = 10            # Active time                                           , in hours per day
        ACTIVE_TIME_days_per_week       = 7             # Active time                                           , in days per week
        self.ACTIVE_TIME_weeks_per_year = 50            # Active time                                           , in weeks per year
        self.FUEL_COST                  = 1.42          # Constant fuel price conversion cost                   , in USD per gallon
        self.LOAD_FACTOR                = 0.8           # Average passenger load factor                         , per flight
        self.HUB                        = 'LFPG'        # Define the hub airport                     

        self.BLOCK_TIME = ACTIVE_TIME_hrs_per_day * ACTIVE_TIME_days_per_week * self.ACTIVE_TIME_weeks_per_year * self.simulation_time_span     # Block Time; total simulation active time     , in hours 

        self.reset()

    def reset(self):
        ''' Define/obtain parameters '''
        # Define Classes
        self.Network = Network(self.network_size)                       # Defines the network; contains airport information; and a function to compute distance between airports
        self.Demand  = Demand(self.network_size, self.simulation_time_span, self.demand_scale, self.demand_stdv_scale, self.demand_average_network_growth)  # Defines the initial demand for the network; and contains a function to generate a new random demand
        self.Fleet   = Fleet(self.number_of_aircraft_types)             # Defines an empty fleet that will be added by running the MILP fleet optimiser; contains aircraft characteristics; and functions to adjust the fleet configuration

        # Setup Class variables
        self.optimal_initial_fleet = None                               # optimial initial fleet is added by running self.solve_fleet_optimiser()
        _, self.Fleet.config       = self.solve_fleet_optimiser()       # Add initial optimised fleet to Fleet class
        for i in range(len(self.Fleet.config)):
            number = self.Fleet.config[list(self.Fleet.config.keys())[i]] + self.fleet_randomiser[i]
            self.Fleet.config[list(self.Fleet.config.keys())[i]] = number if number >= 0 else self.Fleet.config[list(self.Fleet.config.keys())[i]]
        self.Fleet.init_config  = self.Fleet.config

        # Define Actor Critic paramaters
        self.action_space = self.setup_action_space()
        self.state_space  = self.setup_state()

    def solve(self, verbal=False):
        ''' Define MILP parameters'''
        # Variable range
        airports = range(self.network_size)
        aircraft = range(self.number_of_aircraft_types)

        # Define HUB airport
        g = np.ones(self.network_size)
        g[self.Network.ICAO == self.HUB] = 0

        m = Model('Network and Fleet Development')

        # Decision variables
        x = {}
        w = {}
        z = {}
        # Other variables
        s = {}
        sp = {}
        TAT = {}
        R = {}
        RunAC = {}

        ''' Add variables '''
        # Add variables that are not in the objective function
        for k in aircraft:
            type = self.Fleet.types[k]

            s[k]     = self.Fleet.info[type].seats
            sp[k]    = self.Fleet.info[type].speed
            TAT[k]   = self.Fleet.info[type].TAT
            R[k]     = self.Fleet.info[type].max_range
            RunAC[k] = self.Fleet.info[type].rnw_req

        # Add variables that are in the objective function as well
        for i in airports:
            for j in airports:
                if i!=j:
                    origin = self.Network.ICAO[i]    # To check the current airport origin
                    destin = self.Network.ICAO[j]    # To check the current airport destination

                    distance = self.Network.compute_distance(origin, destin)

                    x[i,j] = m.addVar(obj = (5.9*distance**(-0.76) + 0.043)*distance ,lb=0, vtype=GRB.INTEGER, name="x[%s,%s]" % (i, j))
                    w[i,j] = m.addVar(obj = (5.9*distance**(-0.76) + 0.043)*distance ,lb=0, vtype=GRB.INTEGER, name="w[%s,%s]" % (i, j))

                    # Iterate over AC types
                    for k in aircraft:
                        type = self.Fleet.types[k]

                        cTk = self.Fleet.info[type].time_cost        # time based costs
                        cfk = self.Fleet.info[type].fuel_cost        # fuel cost
                        spk = self.Fleet.info[type].speed            # speed of aircraft
                        CXk = self.Fleet.info[type].fixed_oper_cost  # fixed operating cost

                        z[i,j,k] = m.addVar(obj = -(0.7 + 0.3*g[i]*g[j]) * (CXk + cTk * distance/spk + cfk*self.FUEL_COST/1.5*distance), lb=0, vtype=GRB.INTEGER, name="z[%s,%s,%s]" % (i, j, k))

        total_lease_cost = 0
        for k in aircraft:
            type = self.Fleet.types[k]

            CLk = self.Fleet.info[type].lease_cost 
            total_lease_cost += CLk * self.ACTIVE_TIME_weeks_per_year * self.simulation_time_span * self.Fleet.config[type] * self.lease_cost_weight

        m.update()

        ''' Set objective function and constraints '''
        m.setObjective(m.getObjective(), GRB.MAXIMIZE)  # The objective is to maximize revenue

        m.addConstrs(x[i,j] + w[i,j] <= self.Demand.values.iloc[i,j] for i in airports for j in airports if i!=j)
        m.addConstrs(w[i,j] <= self.Demand.values.iloc[i,j]*g[i]*g[j] for i in airports for j in airports if i!=j)
        m.addConstrs(x[i,j] + quicksum(w[i,m]*(1-g[j]) for m in airports if m!=i) + quicksum(w[m,j]*(1-g[i]) for m in airports if m!=j) <= quicksum(z[i,j,k]*s[k]*self.LOAD_FACTOR for k in aircraft) for i in airports for j in airports if i!=j)
        m.addConstrs(quicksum(z[i,j,k] for j in airports if i!=j) - quicksum(z[j,i,k] for j in airports if i!=j) == 0 for i in airports for k in aircraft)
        m.addConstrs(quicksum(quicksum((self.Network.compute_distance(self.Network.ICAO[i], self.Network.ICAO[j])/sp[k]+ (TAT[k]) *(1.5-0.5*g[j]))*z[i, j, k] for j in airports if i!=j) for i in airports)  <= self.BLOCK_TIME*self.Fleet.config[self.Fleet.types[k]]  for k in aircraft)
        m.addConstrs(z[i,j,k] <= (10000 if self.Network.compute_distance(self.Network.ICAO[i], self.Network.ICAO[j]) <= R[k] else 0) for i in airports for j in airports for k in aircraft if i!=j)
        m.addConstrs(z[i,j,k] <= (10000 if ((RunAC[k] <= self.Network.airport_info[self.Network.ICAO[i]].rnw_length) and (RunAC[k] <= self.Network.airport_info[self.Network.ICAO[j]].rnw_length)) else 0) for i in airports for j in airports for k in aircraft if i!=j)
        
        m.update()

        ''' Run the optimisation '''
        if verbal == True:
            m.optimize()
            status = m.status

            if status == GRB.Status.UNBOUNDED:
                print('The model cannot be solved because it is unbounded')

            elif status == GRB.Status.OPTIMAL or True:
                profit = m.objVal - total_lease_cost
                print('***** RESULTS ******')
                print('\nObjective Function Value: \t %g' % (profit))

            elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % status)
        
        elif verbal == False:
            m.Params.LogToConsole = 0
            m.optimize()
            profit = m.objVal - total_lease_cost
        
        return profit

    def solve_fleet_optimiser(self, verbal=False):
        ''' Define MILP parameters'''
        # Variable range
        airports = range(self.network_size)
        aircraft = range(self.number_of_aircraft_types)

        # Define HUB airport
        g = np.ones(self.network_size)
        g[self.Network.ICAO == self.HUB] = 0

        m = Model('Network and Fleet Development')

        # Decision variables
        x = {}
        w = {}
        z = {}
        AC = {}
        # Other variables
        s = {}
        sp = {}
        TAT = {}
        R = {}
        RunAC = {}

        ''' Add variables '''
        # Add variables that are not in the objective function
        for k in aircraft:
            type = self.Fleet.types[k]

            s[k]     = self.Fleet.info[type].seats
            sp[k]    = self.Fleet.info[type].speed
            TAT[k]   = self.Fleet.info[type].TAT
            R[k]     = self.Fleet.info[type].max_range
            RunAC[k] = self.Fleet.info[type].rnw_req

        # Add variables that are in the objective function as well
        for i in airports:
            for j in airports:
                if i!=j:
                    origin = self.Network.ICAO[i]    # To check the current airport origin
                    destin = self.Network.ICAO[j]    # To check the current airport destination

                    distance = self.Network.compute_distance(origin, destin)

                    x[i,j] = m.addVar(obj = (5.9*distance**(-0.76) + 0.043)*distance*2 ,lb=0, vtype=GRB.INTEGER, name="x[%s,%s]" % (i, j))
                    w[i,j] = m.addVar(obj = (5.9*distance**(-0.76) + 0.043)*distance*2 ,lb=0, vtype=GRB.INTEGER, name="w[%s,%s]" % (i, j))

                    # Iterate over AC types
                    for k in aircraft:
                        type = self.Fleet.types[k]

                        cTk = self.Fleet.info[type].time_cost           # time based costs
                        cfk = self.Fleet.info[type].fuel_cost           # fuel cost
                        spk = self.Fleet.info[type].speed               # speed of aircraft
                        CXk = self.Fleet.info[type].fixed_oper_cost     # fixed operating cost

                        z[i,j,k] = m.addVar(obj = -(0.7 + 0.3*g[i]*g[j]) * (CXk + cTk * distance/spk + cfk*self.FUEL_COST/1.5*distance), lb=0, vtype=GRB.INTEGER, name="z[%s,%s,%s]" % (i, j, k))

        for k in aircraft:
            type = self.Fleet.types[k]

            CLk   = self.Fleet.info[type].lease_cost                    # lease cost
            AC[k] = m.addVar(obj = -CLk * self.ACTIVE_TIME_weeks_per_year * self.simulation_time_span * self.lease_cost_weight, lb=0, vtype=GRB.INTEGER, name="AC[%s]" % (k))

        m.update()

        ''' Set objective function and constraints '''
        m.setObjective(m.getObjective(), GRB.MAXIMIZE)  # The objective is to maximize revenue

        m.addConstrs(x[i,j] + w[i,j] <= self.Demand.values.iloc[i,j] for i in airports for j in airports if i!=j)
        m.addConstrs(w[i,j] <= self.Demand.values.iloc[i,j]*g[i]*g[j] for i in airports for j in airports if i!=j)
        m.addConstrs(x[i,j] + quicksum(w[i,m]*(1-g[j]) for m in airports if m!=i) + quicksum(w[m,j]*(1-g[i]) for m in airports if m!=j) <= quicksum(z[i,j,k]*s[k]*self.LOAD_FACTOR for k in aircraft) for i in airports for j in airports if i!=j)
        m.addConstrs(quicksum(z[i,j,k] for j in airports if i!=j) - quicksum(z[j,i,k] for j in airports if i!=j) == 0 for i in airports for k in aircraft)
        m.addConstrs(quicksum(quicksum((self.Network.compute_distance(self.Network.ICAO[i], self.Network.ICAO[j])/sp[k]+ (TAT[k]) *(1.5-0.5*g[j]))*z[i, j, k] for j in airports if i!=j) for i in airports)  <= self.BLOCK_TIME*AC[k] for k in aircraft)
        m.addConstrs(z[i,j,k] <= (10000 if self.Network.compute_distance(self.Network.ICAO[i], self.Network.ICAO[j]) <= R[k] else 0) for i in airports for j in airports for k in aircraft if i!=j)
        m.addConstrs(z[i,j,k] <= (10000 if ((RunAC[k] <= self.Network.airport_info[self.Network.ICAO[i]].rnw_length) and (RunAC[k] <= self.Network.airport_info[self.Network.ICAO[j]].rnw_length)) else 0) for i in airports for j in airports for k in aircraft if i!=j)
        
        m.update()

        ''' Run the optimisation '''
        if verbal == True:
            m.optimize()
            status = m.status

            if status == GRB.Status.UNBOUNDED:
                print('The model cannot be solved because it is unbounded')

            elif status == GRB.Status.OPTIMAL or True:
                profit = m.objVal
                print('***** RESULTS ******')
                print('\nObjective Function Value: \t %g' % (profit))

            elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % status)
            
            fleet = {}
            print("\nFleet configuration:")
            for i in range(len(AC)):
                type = 'type' + str(i+1)
                fleet[type] = int(AC[i].x)
                print(f"{type}: {fleet[type]}")
        
        elif verbal == False:
            m.Params.LogToConsole = 0
            m.optimize()
            profit = m.objVal
        
            fleet = {}
            for i in range(len(AC)):
                type   = 'type' + str(i+1)
                number = int(AC[i].x)
                fleet[type] = number
        
        self.optimal_initial_fleet = fleet

        return profit, fleet

    def step(self, action_index):
        # Run baseline problem
        self.Demand.values = self.Demand.random()           # Generate and update new random demand
        baseline_profit    = self.solve()                   # Solve baseline MILP problem with old fleet and new demand

        # Perform action to fleet
        action = {}
        if action_index < (self.action_space.size - 1):
            action['type']   = self.Fleet.types[action_index//(self.max_number_of_actions * 2)]
            action['number'] = self.action_space[action_index]

            self.Fleet.config, invalid_action = self.Fleet.take_action(action['type'], action['number'])    # update fleet configuration
            penalty = self.fleet_penalty if invalid_action else 0

        elif action_index == (self.action_space.size - 1):
            invalid_action = False
            action         = 'no action'
            penalty        = 0

        new_profit = self.solve()                           # Solve new MILP problem with new fleet and new demand

        # Define outputs; new_state and reward
        new_state = self.setup_state()
        reward    = (new_profit - baseline_profit) / (baseline_profit) + penalty
        
        print(f"{np.round(new_profit, 2)} \t- {np.round(baseline_profit, 2)} \t(diff = {np.round(new_profit - baseline_profit, 2)}) \t---- {np.round(reward, 5)} \t---- fleet: {self.Fleet.config}, \taction: {action}")

        return new_state, reward, invalid_action, new_profit, baseline_profit
    
    def setup_action_space(self):
        action_space_per_type = np.arange(-self.max_number_of_actions, self.max_number_of_actions + 1)
        action_space_per_type = action_space_per_type[action_space_per_type != 0]

        total_action_space = np.array([])
        for i in range(self.number_of_aircraft_types):
            total_action_space = np.append(total_action_space, action_space_per_type, axis=0)
        
        action_space = np.append(total_action_space, 0).astype('int')

        return action_space

    def setup_state(self):
        # State 1 - Airline Fleet
        state_fleet = np.array([])
        for type in self.Fleet.config:
            state_fleet = np.append(state_fleet, self.Fleet.config[type]).astype('float64')
            # state_fleet = np.append(state_fleet, self.Fleet.config[type]/self.Fleet.init_config[type]).astype('float64')

        # State 2 - Demand
        demand_diff  = self.Demand.values.to_numpy() - self.Demand.old_values.to_numpy()
        # demand_diff  = preprocessing.normalize(demand_diff)        # normalisation of demand
        r, c         = np.triu_indices(demand_diff.shape[0], 1)    # Get upper triangle demand values, without the diagonal values
        state_demand = demand_diff[r,c].astype('float64')
    
        # Append both states to single state parameter
        state = np.append(state_fleet, state_demand)

        return state

