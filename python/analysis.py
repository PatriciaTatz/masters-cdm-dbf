from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from scipy.special import erfc

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

    def symbol_sampler(
        self,
        signal: SignalT,
        dt: float,
    ) -> SignalT:
        sps = int(1/(self.conf.rb*dt))
        symbols_at_rb = signal[::sps]
        return symbols_at_rb

    def sim(self, maxSims, maxSNR):

        self.snr_range = np.arange(0,maxSNR, 1)
        self.ber_curve = []

        snr_linear = 10**(self.snr_range / 10) #compute from dB to linear scale to put into erfc 
        self.ber_theory = 0.5 * erfc(np.sqrt(snr_linear))
        
        arrpath = 0 
        txclean = self.signal.elements
        txsymbols = self.symbol_sampler(txclean, self.signal.dt)
        #BER simulation 
        for i, snr in enumerate(self.snr_range):
            print("snr=",snr)

            ber_sum = 0
            for txidx in range(maxSims):

                #Add noise to the transmit signal and it needs to be done for each independent sim 
                self.signal.elements = self.signal.add_awgn_to_elems(txclean, SNR=snr)

                #setup rx, note this simulates for all path 
                self.rx.setup(self.signal)

                #run cdm for all paths 
                _ , recovered_signal = self.rx.assemble() 

                #sample at symbole rate for arrpath only 
                rxsymbols = self.symbol_sampler(
                    recovered_signal.signals[arrpath], 
                    1/self.conf.fs 
                )
 
                """ 
                plt.figure()
                plt.plot(sig.input, '-o', label='Original Rx Signal')
                plt.plot(sig.Despread[arrpath][0], '--o', label='Decorrelated Signal')
                plt.title(f"SNR={snr}")
                plt.legend()
                plt.grid()
                plt.show()
                print(rxbits) 
                """

                #get bit error rate per sim, so this is no false matches / total bits 
                ber_sum += np.mean(txsymbols!=rxsymbols)

            #average all the ber sum for each sim, so this is bersum divided by max sims 
            self.ber_curve.append(ber_sum / maxSims)
        return self.snr_range, self.ber_curve, self.ber_theory

    def plot_curve(self,snr_range, ber_curve):
    # Plot BER curve
        fig, ax = plt.subplots(figsize=(7,5))
        ax.semilogy(
            self.snr_range,
            self.ber_curve,
            marker='o',
            linewidth=2,
            markersize=6,
            label='Simulation'
        )
             
        ax.semilogy(
            self.snr_range,
            self.ber_theory,
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
        if show_plot=="on": plt.show()




if __name__ == "__main__":
    #configure
    Rb = 1
    Tb = 1/Rb
    symbols = 100
    totaltime = symbols*Tb

    config_all_codes = CDMcfg(
        rb=Rb,
        lc=16,
        sps_chip=2,
        alpha=0.5,
        simtime=totaltime,
        no_elem=5
    )

    WHcodes = CodeSelector(config_all_codes)
    weights = WHcodes.generate_weights()
    best, worst  = WHcodes.find_best_clique(
        weights, 
        config_all_codes.no_elem
        )

    config_best_codes = CDMcfg(
        rb=Rb,
        lc=16,
        sps_chip=2,
        alpha=0.5,
        simtime=totaltime,
        no_elem=5,
        codeids=best
    )

    config_worst_codes = CDMcfg(
        rb=Rb,
        lc=16,
        sps_chip=2,
        alpha=0.5,
        simtime=totaltime,
        no_elem=5,
        codeids=worst
    )


    #create signals based on all codes but note it does not use it 
    signal = DiscreteSignal(config_all_codes)

    #run sim with config for best and config for worst for the same signal 
    ber_best = BER(config_best_codes, signal)
    ber_worst = BER(config_worst_codes, signal)

    snr_range, best_ber_curve, ber_theory = ber_best.sim(10**3, 5)
    snr_range, worst_ber_curve, ber_theory = ber_worst.sim(10**3, 5)

    fig, ax = plt.subplots(figsize=(7,5))

    ax.semilogy(
        snr_range,
        best_ber_curve,
        marker='o',
        linewidth=2,
        markersize=6,
        label='Simulation, Best Code Pair'
    )
    ax.semilogy(
        snr_range,
        worst_ber_curve,
        marker='o',
        linewidth=2,
        markersize=6,
        label='Simulation, Worst Code Pair'
    )       
    ax.semilogy(
        snr_range,
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
    plt.show()