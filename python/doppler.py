"""
To calculate max angle and doppler shift a SSO Leo Comms satellite
"""

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file

#load TLE 
#I've chosen LEO Vantage 1 which is an SSO LEO Comms satellite
norad_id = 43113
base = 'https://celestrak.org/NORAD/elements/gp.php'
url = base + f'?CATNR={norad_id}&FORMAT=TLE'
filename = f'tle{norad_id}.txt'
satellite_file = load.download(url, filename=filename)

ts = load.timescale()
LEOVantage1 = load.tle_file(f'tle{norad_id}.txt', ts)[0]

#Looking at Svalbard ground station 
svalbard = wgs84.latlon(+78.22875, +15.39964)

#search time frame in the past  
t0 = ts.utc(2026, 6, 3) 
t1 = ts.utc(2026, 6, 4)

#find events of rise 
t, events = LEOVantage1.find_events(svalbard, t0, t1, altitude_degrees=30.0)
event_names = 'rise above 30°', 'culminate', 'set below 30°'


for ti, event in zip(t, events):
    name = event_names[event]
    print(ti.utc_strftime('%Y %b %d %H:%M:%S'), name)


#Calculate S-curve for maximum elevation angles

passtime = ts.utc(2026, 6, 3, 00, 29,  np.arange(5* 60)) #year, month, day, hour, minute, second 
difference = LEOVantage1 - svalbard
pos = difference.at(passtime)
_, _, sat_range, _, _, sat_range_rate = pos.frame_latlon_and_rates(svalbard)

#Doppler shift
freq_tx = 40e9 #Sband, 3GHz 
c = 299792.458 #speed of light kms
xtime = passtime.utc_datetime()
normalised_doppler = -sat_range_rate.km_per_s / c

plt.figure()
plt.grid()
plt.plot(xtime, normalised_doppler, 'o--')
plt.show()
