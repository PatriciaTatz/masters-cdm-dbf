from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.special import erfc
import copy 
from dataclasses import replace


from codeanalysis import CodeSelector
from config import CDMcfg
from receiver import Receiver, PlotType
from signals import DiscreteSignal


SignalT = np.ndarray[tuple[int], np.dtype[np.float64]]
class BER:
    def __init__(self, config, signal):
        self.conf = config 
        self.rx = Receiver(self.conf)
        self.signal = signal

    def sim(self, maxbits, maxSNR):
        #Directly from the random generator in SigGen
        txsym = np.where(self.signal.txsymbols > 0, 1, -1)                              #Tx Symbols
        arrpath = 0                                                                     #Element Number 
        #Convert SNR from dB to linear ratio in prep for variance formula 
        EbN0dBs = np.arange(0, maxSNR)
        #Calculate enery per bit (Eb) of your signal 
        element_signals = self.signal.elements.copy()                                   #Original transmitted signals
        
        #BER simulation results
        ber_curves = [] 
        for i, EbN0dB in enumerate(EbN0dBs):
            print("snr=",EbN0dBs[i])

            ber_k = np.zeros(self.conf.no_elem)
            total_error = np.zeros(self.conf.no_elem)
            total_bits_tx = np.zeros(self.conf.no_elem)
            while total_bits_tx[0] < maxbits:
                gamma = 10**(EbN0dB/10)    #SNR to linear scale
                rx_noisy = []
                for k in range(self.conf.no_elem):
                    s = element_signals[k]
                    P = np.sum(np.abs(s)**2) / len(s) #Power in array size
                    L = len(s)/len(txsym)
                    Eb = P*L
                    N0 = Eb / gamma #Noise spectral density 
                    n = np.sqrt(N0/2)*np.random.standard_normal(s.shape) #noise array
                    rx = s+n
                    rx_noisy.append(rx)

                #overwrite the elements in the signal object to noisy signal     
                self.signal.elements = rx_noisy

                #setup rx, note this simulates for all path 
                self.rx.setup(self.signal)

                #run cdm for all paths 
                recoveredSignal = self.rx.assemble() 

                #sample at symbol rate for arrpath only 
                rxsym = recoveredSignal.symbols

                #hard decisions
                
                for k in range(self.conf.no_elem):
                    rxsym[k] =  np.where(rxsym[k] > 0, 1, -1)   

                    #calculate errors
                    errors = np.sum(rxsym[k]!=txsym)

                    #update counters
                    total_error[k] += errors
                    total_bits_tx[k] += len(txsym)
            
            ber_k = total_error / total_bits_tx
            ber_curves.append(ber_k)
        ber_avg = [sum(ber_k) / len(ber_k) for ber_k in ber_curves]
        return ebno_db, ber_curves, ber_avg

def plot_curve(ebno_db, ber_array, ber_titles):
    fileloc = f"./results/GCWH/raw_data/"
    #np.save(f"{fileloc}/ebno.npy", ebno_db)
    plt.figure()

    
    #np.save(f"{fileloc}/theory.npy", ber_array[0])
    cmap = plt.get_cmap('turbo')
    n_lines = len(ber_titles)-1
    for i in range(1,len(ber_titles)):
        color = cmap(i/(n_lines))

        np.save(f"{fileloc}/{ber_title[i]}",ber_array[i])
        plt.semilogy(
            ebno_db,
            ber_array[i],
            linestyle='-',
            marker='o',
            linewidth=1,
            markersize=2,
            color=color,
            label=ber_titles[i]
        )

    plt.semilogy(
        ebno_db,
        ber_array[0],
        linestyle='-',
        color='black',
        marker='*',
        linewidth=2,
        markersize=6,
        label=ber_titles[0]
    )
    # Show only 10^n labels
    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())

    plt.xlabel("Eb/No (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title(f"Avg BER for all paths, N=7 with Lc=8")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()


if __name__ == "__main__":
    #configure
    Rb = 30e6
    Tb = 1/Rb
    symbols = 1000
    totaltime = symbols*Tb
    
    lc = 8
    Nelem = [7]
    fam="GCWH"
    print("running=", fam)
    mod = "BPSK"

    ber_title = ['Theoretical'] 
    SNR =  11
    ebno_db = np.arange(0, SNR)
    ebno_linear = 10**(ebno_db/10)
    ber_theory = 0.5*erfc(np.sqrt(ebno_linear))
    ber_data = [ber_theory]

    #Note: when doing WH codes, important for codeids to start from 1 to avoid DC offset code idx=0
    for nelem in Nelem:
        baseconfig = CDMcfg(
            fb=Rb,
            sps_chip=2,
            alpha=0.5,
            simtime=totaltime,
            no_elem=int(nelem),
            lc=lc,
            codeids=np.arange(1,lc),
            CALIBRATE=False,
            CARRIER=True,
            codefam=fam,
            modType=mod
            )
        WHcodes = CodeSelector(baseconfig)
        weights = WHcodes.generate_weights()
        best, worst  = WHcodes.find_best_clique(
                    weights, 
                    baseconfig.no_elem
                    )
        
        config_best = replace(baseconfig, codeids=best)
        config_worst = replace(baseconfig, codeids=worst)
        signal = DiscreteSignal(baseconfig)

        ber_best = BER(config_best, signal)
        ber_worst = BER(config_worst,signal)

        print(f"N={nelem} with best code pairs={best}")
        _, ber_curves, ber_avg  = ber_best.sim(10e2, SNR)                       #BER SIMULATION
        ber_k_0  = [ber_curve[0] for ber_curve in ber_curves]

        ber_title.append(f"N={nelem}_best")
        ber_data.append(ber_avg)

        #ber_title.append(f"N={nelem}")
        #ber_data.append(np.array(ber_k_0))

        #_, ber_curves, ber_avg = ber_worst.sim(10, SNR)
        #ber_title.append(f"N_{nelem}, worst")
        #ber_data.append(ber_avg)
    
    plot_curve(ebno_db, ber_data, ber_title)
    plt.show()
    