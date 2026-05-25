"""
analyse code sequences and find optimal set
assuming that its true Lc  = Narrays * 2 
"""

import components as rf 
from config import *
import measure as msr 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.fft import fft, fftfreq


#codes================
dbfconf = dbfCfg(Lc=4, code="WH", no_arrays=1)   
code = rf.CodeGenerator(dbfconf)
codes = [code.generate(idx) for idx in np.arange(dbfconf.Lc)] #generate all possible codes 

#autocorrelation=========================================
acfs = []
corr = msr.Correlation()
for j in range(len(codes)):
    acf, lag, pkwidth = corr.acf(codes[j], corr_len=dbfconf.Lc)
    acfs.append(acf)
n = len(acfs)
rows = int(np.ceil(np.sqrt(n)))
cols = int(np.ceil(n / rows))
fig_acf, ax_acf = plt.subplots(rows, cols, figsize=(12, 6), sharex=True)
ax_acf = ax_acf.ravel()
""" for k, acf in enumerate(acfs):
    ax_acf[k].plot(lag, acf)
    ax_acf[k].set_title(f"ACF of Code {k}")
 """

#crosscorrelation=========================================
ccfs = []
codesmixed = []
iidx = []
jidx = []
for i in range(len(codes)):
    for j in range(i+1, len(codes)):
        ccfs.append(corr.ccf(codes[i], codes[j]))
        codesmixed.append(codes[i]*codes[j])
        iidx.append(i)
        jidx.append(j)


#cross spectrum density 
#ccfs = codesmixed
fft_ccfs = [fft(ccf) for ccf in ccfs]
freq_bins = fftfreq(dbfconf.Lc) 


#plot of the cross spectrum density 
maxplots=16
n = len(ccfs)
rows = int(np.ceil(np.sqrt(n)))
cols = int(np.ceil(n / rows))
goodpairs = []

for start in range(0, n, maxplots):
    fig_csd, ax_csd = plt.subplots(4, 4, figsize=(12, 6), sharex=True)
    ax_csd = ax_csd.ravel()
    for k in range(start, min(start + maxplots, n)):
        idx = k - start
        ax_csd[idx].stem(freq_bins, np.abs(fft_ccfs[k]))
        ax_csd[idx].set_title(f"fft(c{iidx[k]+1}*c{jidx[k]+1})")

        if np.abs(fft_ccfs[k][1])<=0:
            #any pairs that do not have a peak at 1/Lc
            goodpairs.append([iidx[k]+1, jidx[k]+1])
            ax_csd[idx].title.set_color('green')

        #add grid lines 
        for m in range(-dbfconf.Lc//2, dbfconf.Lc//2 + 1):
            ax_csd[idx].axvline(m / dbfconf.Lc, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
 
    plt.tight_layout(pad=1.0)
    plt.savefig(f"./results/{dbfconf.code}/csd_{start//16}.png")
    plt.close(fig_csd)




