import numpy as np
from scipy.special import erfc
from receiver import Receiver
from config import cdmCfg, signalCfg
import mysignal as s
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogLocator, LogFormatterMathtext

class BER:
    def __init__(self, signalconfig, codeconfig, arrpath):
        self.sigcfg = signalconfig
        self.codecfg = codeconfig
        self.arrpath = arrpath


    def curve(self, maxSims, maxSNR):
        self.snr_range = np.arange(0,maxSNR, 1)
        self.ber_theory = 0.5 * erfc(np.sqrt(self.snr_range))
        self.ber_curve = [] 
        rx = Receiver(self.codecfg)

        for snr in self.snr_range:
            ber_sum = 0
            print("snr=",snr)
            #generate signal 
            for _ in range(maxSims):
                inputSignal = s.DiscreteSignal(self.sigcfg, channel="withNoise", SNR=snr)     
                rx.assemble(inputSignal) 

                txbits = np.sign(inputSignal.arrSignals)
                rxbits = np.sign(rx.arraySignals.DecisionSymbols[self.arrpath])    

                ber_sum += np.mean(txbits!=rxbits)
        
            self.ber_curve.append(ber_sum/maxSims)

        return self.snr_range, self.ber_curve, self.ber_theory

    def plot_curve(self,show_plot):
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
        plt.tight_layout()
        if show_plot=="on": plt.show()


if __name__ == "__main__":
    #configure
    codeCfg = cdmCfg(Lc=32, code="WH") 
    sigCfg = signalCfg(Ts=1, modType="BPSK", Ls=10, no_arrays=2)

    #BER simulation
    ber = BER(sigCfg, codeCfg, arrpath=0)
    ber.curve(maxSims=10**4, maxSNR=20)
    ber.plot_curve(show_plot="on")
