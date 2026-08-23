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

    def symbol_sampler(self, time, signal, dt):
        sps = int(1 / (self.conf.rb * dt)) 
        symbols_at_rb = signal[sps::sps]

        """
        #plot for where the sampler is occuring
        plt.figure()
        plt.title(f"Symbol sampler")
        plt.plot(
            self.signal.time,
            self.signal.transmit, 'blue', linewidth=3, label="Tx Original")
        plt.plot(
            time,
            signal, 'blue', label="Signal")

        plt.stem(
            time[sps::sps],
            symbols_at_rb, 'red', label="Sig Sampled")     
        
        plt.stem(
            self.signal.time,
            self.signal.rrcsymbols, 'orange', label="Tx Symbols")
        
        plt.legend()
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        
        plt.show()
        """
        #BPSKDETECTOR, not SYM TO BITS, wrong name. 
        bits = np.where(symbols_at_rb > 0, 1, -1) 
        return bits

    def sim(self, maxSims, maxSNR):
        ebno_db = np.arange(0, maxSNR)
        ebno_linear = 10**(ebno_db/10)
        
        txbits = np.where(self.signal.txsymbols > 0, 1, -1)
        
        arrpath = 0
        signal_len = len(self.signal.transmit)
        signal = self.signal.transmit.copy()
        eb = np.sum(signal**2) / len(txbits)

        #BER simulation 
        ber_curve = []
        target_error = 100000
        maxbits = 10**7
        for i, ebno in enumerate(ebno_linear):
            print("snr=",ebno_db[i])

            #calculate AWGN noise 
            noise_std = np.sqrt(eb/ (2*ebno))

            totalerrors = 0
            totalbits = 0
            

            for txidx in range(maxSims):
                
                noise = []
                for i in range(self.conf.no_elem):
                    noise_arr = np.random.randn(signal_len)
                    noise.append(noise_std*noise_arr)

                self.signal.add_element_noise(signal, noise)
                
                #setup rx, note this simulates for all path 
                self.rx.setup(self.signal)

                #run cdm for all paths 
                decorr_signal = self.rx.assemble() 

                #sample at symbole rate for arrpath only 
                rxbits = self.symbol_sampler(
                    decorr_signal.time,
                    decorr_signal.signals[arrpath], 
                    1/decorr_signal.fs
                )  
                
                #TEST MF ONLY: SET CALIBRATE=TRUE in Config 

                #get bit error rate per transmission, so this is no false match
                # es / total bits 
                totalerrors += np.sum(txbits!=rxbits)
                totalbits += len(txbits)

                if totalerrors >=target_error: break
                if totalbits >= maxbits: break 

            #average all the ber sum for each sim, so this is bersum divided by max sims 
            ber_curve.append(totalerrors / totalbits)
        return ebno_db, ber_curve

def plot_curve(ebno_db, ber_array, ber_titles):
    fileloc = "./results/WH/raw_data/"
    #np.save(f"{fileloc}/ebno.npy", ebno_db)
    plt.figure()
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
    #np.save(f"{fileloc}/theory.npy", ber_array[0])

    for i in range(1,len(ber_titles)):
        np.save(f"{fileloc}/{ber_title[i]}",ber_array[i])
        plt.semilogy(
            ebno_db,
            ber_array[i],
            linestyle='--',
            marker='o',
            linewidth=2,
            markersize=6,
            label=ber_titles[i]
        )

    # Show only 10^n labels
    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())

    plt.xlabel("Eb/No (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title(f"Element Path=1 with Lc=4")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()


if __name__ == "__main__":
    #configure
    Rb = 6.25e6
    Tb = 1/Rb
    symbols = 1000
    totaltime = symbols*Tb
    
    #PLOT OF INCREASING NC with LPF ON LC=[1,32]
    Nelem = [2]
    ber_title = ['Theoretical'] +[f"N={str(x)}" for x in Nelem]
    SNR =  3
    ebno_db = np.arange(0, SNR)
    ebno_linear = 10**(ebno_db/10)
    ber_theory = 0.5*erfc(np.sqrt(ebno_linear))

    ber_data = [ber_theory]
    #run sim with config for best and config for worst for the same signal 
    lc = 4
    for nelem in Nelem:
        baseconfig = CDMcfg(
            rb=Rb,
            sps_chip=2,
            alpha=0.5,
            simtime=totaltime,
            no_elem=nelem,
            lc=lc,
            codeids=np.arange(1,lc),
            CALIBRATE=True
            )
        WHcodes = CodeSelector(baseconfig)
        weights = WHcodes.generate_weights()
        best, _  = WHcodes.find_best_clique(
                    weights, 
                    baseconfig.no_elem
                    )
        print(f"N={nelem} with code pairs={best}")
        config_n = replace(baseconfig, codeids=best)
        signal = DiscreteSignal(config_n)
        ber = BER(config_n, signal)
        _, curve = ber.sim(10**7, SNR)
        ber_data.append(curve)
    
    plot_curve(ebno_db, ber_data, ber_title)
    plt.show()
    