"""
To calculate max angle and doppler shift a SSO Leo Comms satellite
"""

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load, wgs84

class Doppler:
    def __init__(self):
        #load TLE 
        #I've chosen LEO Vantage 1 which is an SSO LEO Comms satellite
        self.norad_id = '44744'          
        base = 'https://celestrak.org/NORAD/elements/gp.php'
        url = base + f'?CATNR={self.norad_id}&FORMAT=TLE'
        filename = f'tle{self.norad_id}.txt'


    def calculate_shifts(self):

        ts = load.timescale()
        satellite = load.tle_file(f'tle{self.norad_id}.txt', ts)[0]

        #Looking at WSA SSC ground station 
        ssc = wgs84.latlon(-29.05, 115.35)

        #search time frame in the past  
        t0 = ts.utc(2026, 6, 3) 
        t1 = ts.utc(2026, 6, 4)

        #find events of rise 
        t, events = satellite.find_events(
            ssc,
            t0,
            t1,
            altitude_degrees=30.0)

        event_names = 'rise above 30°', 'culminate', 'set below 30°'

        for ti, event in zip(t, events):
            name = event_names[event]
            print(ti.utc_strftime('%Y %b %d %H:%M:%S'), name)


        # Take first complete rise -> culmination -> set
        t_rise = t[0]
        t_culm = t[1]
        t_set = t[2]


        # Create samples across THIS pass
        duration_s = (t_set.tt - t_rise.tt) * 86400
        seconds = np.linspace(0, duration_s, 1000)
        passtime = ts.tt_jd(
        t_rise.tt + seconds / 86400
        )
        difference = satellite - ssc
        pos = difference.at(passtime)
        alt, az, sat_range, alt_rate, az_rate, sat_range_rate = pos.frame_latlon_and_rates(ssc)

        # Doppler
        freq_tx = 40e9
        c = 299792.458 # km/s
        doppler = -(sat_range_rate.km_per_s / c) * freq_tx

        return doppler, alt.degrees, passtime.utc_datetime() 


if __name__ == "__main__":
        
    doppler = Doppler()
    fd_array, elevation, passtime = doppler.calculate_shifts()
    plt.plot(fd_array, elevation)
    plt.show()