class AirportInfo:
    def __init__(self, city, country, lat, lon, rnw_length, slots):
        self.city       = city
        self.country    = country
        self.lat        = lat
        self.lon        = lon
        self.rnw_length = rnw_length
        self.slots      = slots

class FleetInfo:
    def __init__(self, speed, seats, TAT, max_range, rnw_req, lease_cost, fixed_oper_cost, time_cost, fuel_cost):
        self.speed              = speed
        self.seats              = seats
        self.TAT                = TAT               # Average turn around time
        self.max_range          = max_range
        self.rnw_req            = rnw_req
        self.lease_cost         = lease_cost
        self.fixed_oper_cost    = fixed_oper_cost
        self.time_cost          = time_cost
        self.fuel_cost          = fuel_cost

class A2CMemory:
    def __init__(self):
        self.all_rewards        = []
        self.all_actor_loss     = []
        self.all_critic_loss    = []
        self.all_total_loss     = []
        self.all_entropies_loss = []
        self.invalid_actions    = []
    
    def append_loss(self, actor_loss, critic_loss, entropy_loss, total_loss):
        self.all_actor_loss.append(actor_loss)
        self.all_critic_loss.append(critic_loss)
        self.all_entropies_loss.append(entropy_loss)
        self.all_total_loss.append(total_loss)

class A2CEvaluation:
    def __init__(self, state, action, reward, new_profit, baseline_profit):
        self.state           = state.astype('int')
        self.action          = action
        self.reward          = reward
        self.new_profit      = new_profit
        self.baseline_profit = baseline_profit
