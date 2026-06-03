import numpy as np
from scipy.special import erfc
from receiver import Receiver
from config import dbfCfg, signalCfg
from signals import DiscreteSignal
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogLocator, LogFormatterMathtext

class BER:
    def __init__(self, signalconfig, codeconfig):
        self.sigcfg = signalconfig
        self.dbfcfg = codeconfig
        self.rx = Receiver(self.dbfcfg)

    def curve(self, maxSims, maxSNR, sig):
        self.snr_range = np.arange(0,maxSNR, 1)

        snr_linear = 10**(self.snr_range / 10) #compute from dB to linear scale to put into erfc 
        self.ber_theory = 0.5 * erfc(np.sqrt(snr_linear))

        self.ber_curve = []
        txclean = sig.input.copy()
        txsymbols = sig.symbol_sampler(txclean)
        txbits = (txsymbols[0] > 0).astype(int)

        arrpath=0 
        for i, snr in enumerate(self.snr_range):
            print("snr=",snr)
        
            ber_sum = 0
            for txidx in range(maxSims):
                #Add noise to the transmit signal that has been rrc'd filtered 
                sig.input = sig.add_awgn(txclean, SNR=snr)
                sig.create_inputsignals(self.dbfcfg.no_arrays) #based on sig.input
                #CDM4DBF
                self.rx.assemble(sig) 
                #normalise the signal and sample at symbole rate
                rxsymbols = sig.symbol_sampler(sig.Despread[arrpath][0])
 
                rxbits = (rxsymbols[0] > 0).astype(int)
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
                ber_sum += np.mean(txbits!=rxbits)
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
    sigconf = signalCfg(Ts=1, modType="BPSK", Ns=10)  
    dbfconf_best = dbfCfg(Lc=8, code="WH", no_arrays=1, codeids=[6,7])   
    dbfconf_worst = dbfCfg(Lc=8, code="WH", no_arrays=1, codeids=[0,1])   

    #create signal 
    signal = DiscreteSignal(sigconf)

    #preserve memory for processed signals
    ber_best = BER(sigconf, dbfconf_best)
    ber_worst = BER(sigconf, dbfconf_worst)

    snr_range, best_ber_curve, ber_theory = ber_best.curve(10**4, 9, signal)
    snr_range, worst_ber_curve, ber_theory = ber_worst.curve(10**4, 9, signal)


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