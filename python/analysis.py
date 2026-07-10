from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from scipy.special import erfc
import copy 

from codeanalysis import CodeSelector
from config import CDMcfg
from receiver import Receiver
from signals import DiscreteSignal


SignalT = np.ndarray[tuple[int], np.dtype[np.float64]]

class BER:
    def __init__(self, config, signal):
        self.conf = config 
        self.rx = Receiver(self.conf)
        self.signal = signal

    def symbol_sampler(self, time, signal, dt):
        sps = int(round(1 / (self.conf.rb * dt)))
        symbols_at_rb = signal[::sps]
        """
        #plot for where the sampler is occuring
        plt.figure()
        plt.title(f"Symbol sampler")
        plt.plot(
            time,
            signal, 'blue', label="Signal")
        plt.stem(
            time[::sps],
            symbols_at_rb, 'red', label="Sig Sampled")        
        plt.stem(
            self.signal.time[:len(self.signal.rrcsymbols)],
            self.signal.rrcsymbols, 'orange', label="Tx Symbols")
        plt.legend()
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.show()
        """
        bits = np.where(symbols_at_rb > 0, 1, -1)
        return bits

    def sim(self, maxSims, maxSNR):
        #FORTEST
        #ebno_db = np.array([100])
        ebno_db = np.arange(0, maxSNR)

        ebno_linear = 10**(ebno_db/10)
        ber_theory = 0.5*erfc(np.sqrt(ebno_linear))
    
        txbits = self.symbol_sampler(
            self.signal.time,
            self.signal.transmit,
            1/self.conf.fs
        )  
        arrpath = 0 
        signal_len = len(self.signal.transmit)

        #signalpower = np.mean(no.abs(txsignal)**2)

        #BER simulation 
        ber_curve = []
        for i, ebno in enumerate(ebno_linear):
            print("snr=",ebno_db[i])

            #calculate AWGN noise 
            eb = 1
            noise_var = eb / (2*ebno)
            noise_std = np.sqrt(noise_var)
            signal = self.signal.transmit.copy()

            totalerrors = 0
            totalbits = 0
            for txidx in range(maxSims):
                #Add noise to the transmit signal and it needs to be done for each independent transmission tx
                noise = noise_std*np.random.randn(signal_len)
                
                #FORTEST
                #noisysignal = self.signal.transmit + noise
                self.signal.elements = self.signal.create_elementsignals(signal, noise)
                #setup rx, note this simulates for all path 
                self.rx.setup(self.signal)

                #run cdm for all paths 
                spread_signal, despread_signal = self.rx.assemble() 

                #sample at symbole rate for arrpath only 
                rxbits = self.symbol_sampler(
                    despread_signal.time,
                    despread_signal.signals[arrpath], 
                    1/spread_signal.fs
                )  
   
                """
                #TEST: For w/o Rx 
                #sample at symbol rate for arrpath only 
                rxbits = self.symbol_sampler(
                    self.signal.time,
                    noisysignal, 
                    1/self.conf.fs
                )  
                """
                #rxsymbols = txsymbols + noise
                #get bit error rate per transmission, so this is no false matches / total bits 
                totalerrors += np.sum(txbits!=rxbits)
                totalbits += len(txbits)

            #average all the ber sum for each sim, so this is bersum divided by max sims 
            ber_curve.append(totalerrors / totalbits)
        return ebno_db, ber_curve, ber_theory

def plot_curve(ebno_db, ber_best, ber_worst, ber_theory):
# Plot BER curve
    fig, ax = plt.subplots(figsize=(7,5))
    ax.semilogy(
        ebno_db,
        ber_best,
        marker='o',
        linewidth=2,
        markersize=6,
        label='Best Clique'
    )
    ax.semilogy(
        ebno_db,
        ber_worst,
        marker='o',
        linewidth=2,
        markersize=6,
        label='Worst Clique'
    )
    
    ax.semilogy(
        ebno_db,
        ber_theory,
        '--',
        linewidth=2,
        label='Theoretical BPSK BER'
    )  
    
    # Log-scale tick formatting
    ax = plt.gca()

    # Major ticks at powers of 10
    ax.yaxis.set_major_locator(LogLocator(base=10.0))

    # Format as 10^-6 instead of 1e-6
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()


if __name__ == "__main__":
    #configure
    Rb = 1
    Tb = 1/Rb
    symbols = 100
    totaltime = symbols*Tb
    lc = 16
    no_elem = 2

    
    config_all_codes = CDMcfg(
        rb=Rb,
        lc=lc,
        sps_chip=2,
        alpha=0.5,
        simtime=totaltime,
        no_elem=no_elem
    )

    WHcodes = CodeSelector(config_all_codes)
    weights = WHcodes.generate_weights()
    best, worst  = WHcodes.find_best_clique(
        weights, 
        config_all_codes.no_elem
        )

    config_best_codes = CDMcfg(
        rb=Rb,
        lc=lc,
        sps_chip=2,
        alpha=0.5,
        simtime=totaltime,
        no_elem=no_elem,
    )

    config_worst_codes = CDMcfg(
        rb=Rb,
        lc=lc,
        sps_chip=2,
        alpha=0.5,
        simtime=totaltime,
        no_elem=no_elem,
    )

    #create signals based on all codes but note it does not use it 
    best_signal = DiscreteSignal(config_best_codes)
    worst_signal = DiscreteSignal(config_worst_codes)

    #run sim with config for best and config for worst for the same signal 
    ber_best = BER(config_best_codes, best_signal)
    ber_worst = BER(config_worst_codes, worst_signal)

    ebno_db, ber_best_curve, ber_theory = ber_best.sim(10**3, 5)
    ebno_db, ber_worst_curve, _ = ber_worst.sim(10**3, 5)

    plot_curve(ebno_db, ber_best_curve, ber_worst_curve, ber_theory)
    plt.show()