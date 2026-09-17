from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.special import erfc
import copy 
from dataclasses import replace
import itertools
from concurrent.futures import ProcessPoolExecutor
import os 

from codeanalysis import CodeSelector
from config import CDMcfg
from receiver import Receiver, PlotType
from signals import DiscreteSignal


SignalT = np.ndarray[tuple[int], np.dtype[np.float64]]
class BER:
    def __init__(self, config):
        self.conf = config 

    def sim_parallel(self, maxbits, maxSNR):
        EbN0dBs = np.arange(0, maxSNR)

        with ProcessPoolExecutor() as executor:
            args = [
                (snr, self.conf, maxbits)
                for snr in EbN0dBs
            ]

            results = list(executor.map(self.sim_one_snr, args))
            for r in results:
                print(f"SNR={r["snr"]} PID={r["pid"]}")
            ber_curves = [r["ber"] for r in results]

        ber_avg = [np.mean(curve) for curve in ber_curves]
        return ber_curves, ber_avg


    def sim_one_snr(self, args):
        EbN0dB, conf, maxbits = args
        rx = Receiver(conf)
        total_error = np.zeros(conf.no_elem)
        total_bits_tx = np.zeros(conf.no_elem)

        while np.any(total_bits_tx < maxbits):
            #generate signal
            signal = DiscreteSignal(conf)
            txsym = np.where(signal.txsymbols > 0, 1, -1)                              #Tx Symbols

            gamma = 10**(EbN0dB/10)    #SNR to linear scale
            rx_noisy = []
            for k in range(self.conf.no_elem):
                s = signal.elements[k]
                P = np.sum(np.abs(s)**2) / len(s) #Power in array size
                L = len(s)/len(txsym)
                Eb = P*L
                N0 = Eb / gamma #Noise spectral density 
                n = np.sqrt(N0/2)*np.random.standard_normal(s.shape) #noise array
                r = s+n
                rx_noisy.append(r)

            #overwrite the elements in the signal object to noisy signal     
            signal.elements = rx_noisy

            #setup rx, note this simulates for all path 
            rx.setup(signal)

            #run cdm for all paths 
            recoveredSignal = rx.assemble() 

            #sample at symbol rate for arrpath only 
            rxsym = recoveredSignal.symbols
            
            #hard decisions
            for k in range(conf.no_elem):
                rsym =  np.where(rxsym[k] > 0, 1, -1)   

                #calculate errors
                errors = np.sum(rsym!=txsym)

                #update counters - this doesnt allow for rotations 
                total_error[k] += errors
                total_bits_tx[k] += len(txsym)
        ber_k = total_error/total_bits_tx
        return {
            "snr": EbN0dB,
            "pid": os.getpid(),
            "ber":total_error/total_bits_tx
        }

    def sim(self, maxbits, maxSNR):
        #Directly from the random generator in SigGen
        arrpath = 0                                                                     #Element Number 
        #Convert SNR from dB to linear ratio in prep for variance formula 
        EbN0dBs = np.arange(0, maxSNR)
        #Calculate enery per bit (Eb) of your signal 
        maxerrors = 1000 
        #BER simulation results
        ber_curves = [] 

        #initiate the Rx class 
        rx = Receiver(self.conf)

        for i, EbN0dB in enumerate(EbN0dBs):
            print("snr=",EbN0dBs[i])

            ber_k = np.zeros(self.conf.no_elem)
            total_error = np.zeros(self.conf.no_elem)
            total_bits_tx = np.zeros(self.conf.no_elem)

            while (np.any(total_bits_tx < maxbits)):

                #generate signal
                signal = DiscreteSignal(self.conf)
                txsym = np.where(signal.txsymbols > 0, 1, -1)                              #Tx Symbols

                gamma = 10**(EbN0dB/10)    #SNR to linear scale
                rx_noisy = []
                for k in range(self.conf.no_elem):
                    s = signal.elements[k]
                    P = np.sum(np.abs(s)**2) / len(s) #Power in array size
                    L = len(s)/len(txsym)
                    Eb = P*L
                    N0 = Eb / gamma #Noise spectral density 
                    n = np.sqrt(N0/2)*np.random.standard_normal(s.shape) #noise array
                    rx = s+n
                    rx_noisy.append(rx)

                #overwrite the elements in the signal object to noisy signal     
                signal.elements = rx_noisy

                #setup rx, note this simulates for all path 
                rx.setup(signal)

                #run cdm for all paths 
                recoveredSignal = rx.assemble() 

                #sample at symbol rate for arrpath only 
                rxsym = recoveredSignal.symbols
                
                #hard decisions
                for k in range(self.conf.no_elem):
                    rsym =  np.where(rxsym[k] > 0, 1, -1)   

                    #calculate errors
                    errors = np.sum(rsym!=txsym)

                    #update counters - this doesnt allow for rotations 
                    total_error[k] += errors
                    total_bits_tx[k] += len(txsym)
            
            ber_k = total_error / total_bits_tx
            ber_curves.append(ber_k)
        ber_avg = [sum(ber_k) / len(ber_k) for ber_k in ber_curves]
        return ber_curves, ber_avg

def plot_curve(ebno_db, ber_array, ber_titles):
    fileloc = f"./results/{fam}/raw_data/"
    #np.save(f"{fileloc}/ebno.npy", ebno_db)
    plt.rcParams['font.size'] = 15

    
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
    plt.title(f"Avg BER | B = fb/2")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()


if __name__ == "__main__":
    #configure
    Rb = 30e6
    Tb = 1/Rb
    symbols = 1000
    totaltime = symbols*Tb
    
    lc  = 8
    Nelem = np.array([1,2])
    fam="WH"
    print("running=", fam)
    mod = "BPSK"
    stage="BB"

    ber_title = ['Theoretical'] 
    SNR =  6
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
            codeids=np.arange(1,6),
            CALIBRATE=False,
            CARRIER=True,
            SPREADING_STAGE=stage,
            codefam=fam,
            modType=mod
            )
        WHcodes = CodeSelector(baseconfig)
        weights = WHcodes.generate_weights()
        best, worst, ranked  = WHcodes.find_best_clique(
                    weights, 
                    baseconfig.no_elem
                    )
        """
        i=1
        for clique in ranked:
            config_pair = replace(baseconfig, codeids=np.array(clique))
            signal = DiscreteSignal(config_pair)
            print("clique=",clique)
            ber = BER(config_pair)
            ber_curves, ber_avg = ber.sim(1e3,SNR)
            ber_title.append(f"Clique = {clique} | Rank = {i}")
            ber_data.append(ber_avg)
            i+=1
        
        """
        if nelem == 2:
            best = np.array((2,3))
        config_best = replace(baseconfig, codeids=best)
        config_worst = replace(baseconfig, codeids=worst)

        ber_best = BER(config_best)
        #ber_worst = BER(config_worst)

        print(f"N={nelem} with best code pairs={best}")
        #ber_curves, ber_avg  = ber_best.sim(1e5, SNR)                       #BER SIMULATION
        ber_curves, ber_avg = ber_best.sim_parallel(1e2,SNR)
        ber_k_0  = [ber_curve[0] for ber_curve in ber_curves]

        ber_title.append(f"N_{nelem}_Lc_{lc}_best_{stage}")
        ber_data.append(ber_avg)
        
        #ber_title.append(f"N={nelem}")
        #ber_data.append(np.array(ber_k_0))

        #ber_curves, ber_avg = ber_worst.sim(10, SNR)
        #ber_title.append(f"N_{nelem}, worst")
        #ber_data.append(ber_avg)
        
    plot_curve(ebno_db, ber_data, ber_title)
    plt.show()
    