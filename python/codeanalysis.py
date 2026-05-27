"""
analyse code sequences and find optimal set
assuming that its true Lc  = Narrays * 2 
"""

from components import CodeGenerator
from config import dbfCfg


from scipy.signal import find_peaks, peak_widths, correlate, correlation_lags, csd
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.fft import fft, fftfreq
        

class CodeSelector:
    #Method to rank and select codes 
    def __init__(self, config):
        self.Lc = config.Lc
        self.codefam = config.code
        codes = CodeGenerator(config)
        self.allcodes = [codes.generate(idx) for idx in np.arange(self.Lc)]
        self.lags = correlation_lags(self.Lc, self.Lc)


    def acf(self):
        #Find the peak width at zero timelag
        #Rank: narrow peak (best) to wider peak (worst)
        acf_widths = []

        for j in range(len(self.allcodes)):
            #generate lags of equal length 
            acf = correlate(self.allcodes[j], self.allcodes[j])

            peaks, _ = find_peaks(acf)
            print(np.argmax(acf[peaks]))
            print(j)
            try:
                largest_peak_index = peaks[np.argmax(acf[peaks])]
                acf_widths.append(results_half[3]-results_half[2])

                #max width at half max 
                results_half = peak_widths(acf, [largest_peak_index], rel_height=0.5) 
                lagx1, lagx2 = self.lagscaler(results_half)
                plt.plot(self.lags[largest_peak_index] , acf[largest_peak_index ], "x")
                plt.hlines(results_half[1], lagx1, lagx2, color="C2")  #yvalue, xmin, xmax 
        
            except: pass

            #plot and save images 
            plt.plot(self.lags, acf)
            plt.xlabel("Lag")
            plt.ylabel("ACF")
            plt.grid()
            plt.title(f"ACF: code[{j}]")

            plt.savefig(f"./results/{self.codefam}/acf/acf_code{j}.png")
            plt.close()

  

    def ccf(self):
        #Find the CCF peak at zero time lag and classifies them as bad 
        #Rank: narrow peak (best) to wider peak (worst) at zero lag, otherwise ignore 
        # Positive → when one signal increases, the other tends to increase (they move together)
        # Negative → when one signal increases, the other tends to decrease (they move oppositely)        
        ccf_widths = []
        ccfs = []
        csds = []
        pairidx = []
        for i in range(len(self.allcodes)):
            for j in range(i+1, len(self.allcodes)):
                ccf = correlate(self.allcodes[i], self.allcodes[j])
                fbins, csdx = csd(self.allcodes[i], self.allcodes[j], nperseg=self.Lc, return_onesided=False)
                
                csds.append(csdx)
                ccfs.append(ccf)
                pairidx.append((i,j))

                peaks, _ = find_peaks(ccf)
                largest_peak_index = peaks[np.argmax(ccf[peaks])]

            
                #max width at half max 
                results_half = peak_widths(ccf, [largest_peak_index], rel_height=0.5) 
                lagx1, lagx2 = self.lagscaler(results_half)

                ccf_widths.append(results_half[3]-results_half[2])
            
                #plot and save images 
                plt.figure()
                plt.plot(self.lags,ccf)
                plt.plot(self.lags[largest_peak_index] , ccf[largest_peak_index ], "x")
                plt.hlines(results_half[1], lagx1, lagx2, color="C2")  #yvalue, xmin, xmax 
                plt.grid()
                plt.title(f"CCF: code[{i}]xcode[{j}]")
                plt.savefig(f"./results/{self.codefam}/ccf/ccf_codes{i}x{j}.png")
                plt.close()

                plt.figure()
                plt.stem(fbins,np.abs(csdx))
                plt.title(f"CSD: code[{i}]xcode[{j}]")
                for m in range(-self.Lc//2, self.Lc//2 + 1):
                    plt.axvline(m / self.Lc, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  
                k_vals = np.arange(-self.Lc//2, self.Lc//2 + 1, 4 )   # step = 4
                ticks = k_vals / self.Lc
                labels = [rf"{fbins[k]}Lc" for k in k_vals]
                plt.xticks(ticks, labels)
                plt.savefig(f"./results/{self.codefam}/csd/csd_codes{i}x{j}.png")
                plt.close()

        
    def lagscaler(self, results_half):

        left_ip = results_half[2][0]
        right_ip = results_half[3][0]

        left_lag = np.interp(left_ip, np.arange(len(self.lags)), self.lags)
        right_lag = np.interp(right_ip, np.arange(len(self.lags)), self.lags)

        return  left_lag, right_lag 

if __name__ == "__main__":
    #initiate code config
    WHcfg = dbfCfg(Lc=32, code="WH")
    #WHcodes = CodeSelector(WHcfg)
    #WHcodes.acf()
    #WHcodes.ccf()

    Goldcfg = dbfCfg(Lc=31, code="Gold")
    Goldcodes = CodeSelector(Goldcfg)
    Goldcodes.acf()
