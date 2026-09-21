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
        total_bits_tx = 0
        maxerrors = 1000

        while (
                np.any((total_bits_tx < maxbits) &
                    (total_error < maxerrors)
                )
            ):
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
                txsym = np.where(signal.symbols_elements[k] > 0, 1, -1)       #Tx Symbols
                rsym =  np.where(rxsym[k] > 0, 1, -1)   

                #calculate errors
                errors = np.sum(rsym!=txsym)

                #update counters - this doesnt allow for rotations 
                total_error[k] += errors
            total_bits_tx += len(txsym)
        ber_k = total_error/total_bits_tx
        return {
            "snr": EbN0dB,
            "pid": os.getpid(),
            "ber":total_error/total_bits_tx
        }


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
    plt.title(f"Avg BER | WH N=2 Lc=8")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()


if __name__ == "__main__":
    #configure
    Rb = 30e6
    Tb = 1/Rb
    symbols = 1000
    totaltime = symbols*Tb
    
    lc  = 16
    Nelem = np.array([8])
    fam="WH"
    print("running=", fam)
    mod = "BPSK"
    stage="BB"

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
        for clique in ranked[:3]:
            config_pair = replace(baseconfig, codeids=np.array(clique))
            signal = DiscreteSignal(config_pair)
            print("clique=",clique)
            ber = BER(config_pair)
            ber_curves, ber_avg = ber.sim_parallel(1e4,SNR)
            ber_title.append(f"No = {clique} | Rank = {i}")
            ber_data.append(ber_avg)
            i+=1

        """
        config_best = replace(baseconfig, codeids=best)
        #config_worst = replace(baseconfig, codeids=worst)

        ber_best = BER(config_best)
        #ber_worst = BER(config_worst)

        print(f"N={nelem} with best code pairs={best}")
        ber_curves, ber_avg = ber_best.sim_parallel(1e4,SNR)
        ber_k_0  = [ber_curve[0] for ber_curve in ber_curves]

        ber_title.append(f"N_{nelem}_Lc_{lc}")
        ber_data.append(ber_avg)
        
        #ber_title.append(f"N={nelem}")
        #ber_data.append(np.array(ber_k_0))

        #ber_curves, ber_avg = ber_worst.sim(10, SNR)
        #ber_title.append(f"N_{nelem}, worst")
        #ber_data.append(ber_avg)
        
    plot_curve(ebno_db, ber_data, ber_title)
    plt.show()
    