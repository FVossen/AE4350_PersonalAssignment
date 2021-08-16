from math import pi, sin, cos, asin, sqrt
import pandas as pd

from Classes.ClassObjects import AirportInfo

class Network:
    def __init__(self, network_size):
        if network_size > 20:
            raise KeyError('network_size should be less or equal than 20')

        airport_data = pd.read_excel("Data/airport_data.xlsx", index_col = 0, usecols = 'B:V', skiprows = 3, nrows = 6).T
        airport_data = airport_data.iloc[0:network_size, :]

        self.ICAO = airport_data.index.to_numpy()

        self.airport_info = {}
        for ICAO in self.ICAO:
            info = airport_data.loc[ICAO, :]
            self.airport_info[ICAO] = AirportInfo(info['City'], info['Country'], info['Latitude (deg)'], 
                                                  info['Longitude (deg)'], info['Runway (m)'], info['Available slots'])

    def compute_distance(self, origin, destination):
        orig = self.airport_info[origin]
        dest = self.airport_info[destination]

        t1 = pow( sin((orig.lat - dest.lat) / 2 * (pi / 180)) , 2)
        t2 = cos(orig.lat * (pi / 180)) * cos(dest.lat * (pi / 180)) * pow( sin(( orig.lon - dest.lon) / 2 * (pi / 180) ), 2)
        deltaSigma = 2 * asin(sqrt(t1 + t2))
        distance = 6371 * deltaSigma
        return distance

