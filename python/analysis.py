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
        #ebno_db = np.array([10])

        ebno_db = np.arange(0, maxSNR)
        ebno_linear = 10**(ebno_db/10)
    
        txbits = self.symbol_sampler(
            self.signal.time,
            self.signal.transmit,
            1/self.conf.fs
        )  
        arrpath = 0
        signal_len = len(self.signal.transmit)
        signal = self.signal.transmit.copy()

        #signalpower = np.mean(no.abs(txsignal)**2)

        #BER simulation 
        ber_curve = []
        for i, ebno in enumerate(ebno_linear):
            print("snr=",ebno_db[i])

            #calculate AWGN noise 
            eb = 1#from the samples per symbol
            noise_var = eb / (2*ebno)
            noise_std = np.sqrt(noise_var)

            totalerrors = 0
            totalbits = 0
            

            for txidx in range(maxSims):
                #Add noise to the transmit signal and it needs to be done for each independent transmission tx                
                #add noise to each element path 

                #noise_arr = np.random.randn(signal_len)
                #noise = [noise_std*noise_arr]
                #noise.append(np.zeros(len(noise_arr)))
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


                """
                #TEST: For w/o Rx 
                #sample at symbol rate for arrpath only 
                rxbits = self.symbol_sampler(
                    self.signal.time,
                    noisysignal, 
                    1/self.conf.fs
                )  
                #for without filter 
                #rxsymbols = txsymbols + noise
                """

                #get bit error rate per transmission, so this is no false matches / total bits 
                totalerrors += np.sum(txbits!=rxbits)
                totalbits += len(txbits)

            #average all the ber sum for each sim, so this is bersum divided by max sims 
            ber_curve.append(totalerrors / totalbits)
        return ebno_db, ber_curve

def plot_curve(ebno_db, ber_array, ber_titles):

    #save the data f"./results/{self.codefam}/sf/sf_code{self.codeids[j]}.png
    fileloc = "./results/WH/raw_data/"
    np.save(f"{fileloc}/ebno.npy", ebno_db)

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
    np.save(f"{fileloc}/theory.npy", ber_array[0])

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
    symbols = 100
    totaltime = symbols*Tb

    
    #create signal
    """
    #PLOT OF INCREASING LC with LPF ON 
    Lc = [8,16,32]
    ber_title = ['Theoretical'] +[f"Lc={str(x)}" for x in Lc]
    SNR = 5
    ebno_db = np.arange(0, SNR)
    ebno_linear = 10**(ebno_db/10)
    ber_theory = 0.5*erfc(np.sqrt(ebno_linear))

    ber_data = [ber_theory]
    #run sim with config for best and config for worst for the same signal 
    for lc in Lc:

        baseconfig = CDMcfg(
            rb=Rb,
            sps_chip=2,
            alpha=0.5,
            simtime=totaltime,
            no_elem=4,
            lc=lc,
            codeids=np.arange(1,lc)
            )

        WHcodes = CodeSelector(baseconfig)
        weights = WHcodes.generate_weights()
        best, _  = WHcodes.find_best_clique(
                    weights, 
                    baseconfig.no_elem
                    )

        config_lc = replace(baseconfig, lc=lc, codeids=best)
        signal = DiscreteSignal(config_lc)
        ber = BER(config_lc, signal)
        
        _, curve = ber.sim(10**3, SNR)
        
        ber_data.append(curve)
    
    plot_curve(ebno_db, ber_data, ber_title)
    plt.show()
    """

    
    #PLOT OF INCREASING NC with LPF ON LC=[1,32]
    Nelem = [1,5,6,7]
    ber_title = ['Theoretical'] +[f"N={str(x)}" for x in Nelem]
    SNR = 6
    ebno_db = np.arange(0, SNR)
    ebno_linear = 10**(ebno_db/10)
    ber_theory = 0.5*erfc(np.sqrt(ebno_linear))

    ber_data = [ber_theory]
    #run sim with config for best and config for worst for the same signal 

    for nelem in Nelem:
        baseconfig = CDMcfg(
            rb=Rb,
            sps_chip=2,
            alpha=0.5,
            simtime=totaltime,
            no_elem=nelem,
            lc=8,
            codeids=np.arange(1,8)
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
        _, curve = ber.sim(10**3, SNR)
        ber_data.append(curve)
    
    plot_curve(ebno_db, ber_data, ber_title)
    plt.show()
    
    """
    #PLOT OF TOP TEN WORST CODES 
    signal = DiscreteSignal(baseconfig)
    WHcodes = CodeSelector(baseconfig)
    
    weights = WHcodes.generate_weights()
    best, worst  = WHcodes.find_best_clique(
        weights, 
        baseconfig.no_elem
        )
    #top_ten = [(0,3), (0,2), (2,3), (0,1), (1,3), (1,2)]
    top_ten = [(0,3), (1,2)]
    ber_title = ['Theoretical'] +[str(x) for x in top_ten]
    SNR = 2

    ebno_db = np.arange(SNR)
    ebno_linear = 10**(ebno_db/10)
    ber_theory = 0.5*erfc(np.sqrt(ebno_linear))

    ber_data = [ber_theory]
    
    #run sim with config for best and config for worst for the same signal 
    for clique in top_ten:
        config_coded = replace(baseconfig, codeids=list(clique))
        ber = BER(config_coded, signal)
        _, curve = ber.sim(1, SNR)
        ber_data.append(curve)
    
    plot_curve(ebno_db, ber_data, ber_title)
    plt.show()

    
    #PLOT OF BEST vs WORST CLIQUE 
    signal = DiscreteSignal(baseconfig)
    config_best_codes = replace(baseconfig, codeids=best)
    config_worst_codes = replace(baseconfig, codeids=worst)

    print(config_best_codes)
    print(config_worst_codes)

    ber_best = BER(config_best_codes, signal)
    ber_worst = BER(config_worst_codes, signal)
    ebno_db, best_curve = ber_best.sim(noSims, SNR)
    _, worst_curve = ber_worst.sim(noSims,SNR)
   
    SNR = 2
    noSims = 10**3
    ebno_db = np.arange(0, SNR)
    ebno_linear = 10**(ebno_db/10)
    ber_theory = 0.5*erfc(np.sqrt(ebno_linear))

    ber_data = [best_curve, worst_curve, ber_theory]
    ber_titles = ['Best Clique',  'Worst Clique', 'Theoretical BER']
    ber_mrks = ['-o', '-o', '--g']
    ber_best.plot_curve(ebno_db, ber_data, ber_titles, ber_mrks)
    plt.show()

    np.save("ebno", ebno_db)
    np.save("ber_sharedAWGN_4_3_not_normalised", ber_best_curve)
    np.save("ber_theory", ber_theory)


    #PLOTTER
    ebno_range = np.load("ebno.npy")
    ber_shared = np.load("ber_sharedAWGN_4_3_not_normalised.npy")
    ber_indie = np.load("ber_indieAWGN_4_3_not_normalised.npy")
    ber_theory = np.load("ber_theory.npy")
    ber_data = [ber_shared, ber_indie, ber_theory]
    ber_titles = ['Shared AWGN', 'Independent AWGN', 'Theoretical BER']
    ber_mrks = ['-o', '-o', '--g']
    plot_curve(ebno_range, ber_data, ber_titles, ber_mrks)
    plt.show()
    """