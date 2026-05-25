import numpy as np
from scipy.special import erfc
from receiver import Receiver
from config import dbfCfg, signalCfg
from signals import DiscreteSignal, ArraySignals
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogLocator, LogFormatterMathtext

class BER:
    def __init__(self, signalconfig, codeconfig):
        self.sigcfg = signalconfig
        self.dbfcfg = codeconfig
        self.rx = Receiver(self.dbfcfg)

        #BER simulation
        self.signal = DiscreteSignal(self.sigcfg)
        self.array = ArraySignals()
        self.Lss = len(self.signal.transmit) * self.dbfcfg.Lc


    def curve(self, maxSims, maxSNR):
        self.snr_range = np.arange(0,maxSNR, 1)

        snr_linear = 10**(self.snr_range / 10)
        self.ber_theory = 0.5 * erfc(np.sqrt(snr_linear))

        self.ber_curve = np.zeros(len(self.snr_range))        
        arrpath=0 
        for i, snr in enumerate(self.snr_range):
            print("snr=",snr)
            #preserve and reset arrays 
            self.array.create_inputsignals(self.signal, self.dbfcfg.no_arrays)
            self.array.create_rxsignals(self.Lss, self.sigcfg.Ns)

            ber_sum = 0
            for txidx in range(maxSims):
                #Add noise to the same signal 
                txsignal = self.signal.add_awgn(self.signal.transmit, SNR=snr)
                txsymbols = self.signal.symbol_sampler(txsignal)

                #CDM4DBF
                self.rx.assemble(self.array) 
                rxsymbols = self.signal.symbol_sampler(self.array.Despread[arrpath][0])
 
                txbits = np.sign(txsymbols[0])
                rxbits = np.sign(rxsymbols[0])    

                ber_sum += np.mean(txbits!=rxbits)
        
            self.ber_curve[i] = ber_sum / maxSims

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
        if show_plot=="on": plt.show()


if __name__ == "__main__":
    #configure
    sigconf = signalCfg(Ts=5, modType="BPSK", Ns=1)  
    dbfconf = dbfCfg(Lc=5, code="mLength", no_arrays=1)   

    ber = BER(sigconf, dbfconf)
    ber.curve(maxSims=10**4, maxSNR=12)
    ber.plot_curve(show_plot="on")
