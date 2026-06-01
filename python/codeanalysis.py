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
import itertools
from math import isclose

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
        #ACF Widths = [(Amplitude, Code Index J)]
        #Rank: ACF WIDTHS MINIMISED 
        self.acf_widths = []

        for j in range(len(self.allcodes)):
            #generate lags of equal length 
            acf = correlate(self.allcodes[j], self.allcodes[j])
            peaks, _ = find_peaks(acf)

            try:
                largest_peak_index = peaks[np.argmax(acf[peaks])]
                #max width at half max 
                results_half = peak_widths(acf, [largest_peak_index], rel_height=0.5) 
                self.acf_widths.append((j, float(results_half[3][0]-results_half[2][0])))

                lagx1, lagx2 = self.lagscaler(results_half)
                plt.plot(self.lags[largest_peak_index] , acf[largest_peak_index ], "x")
                plt.hlines(results_half[1], lagx1, lagx2, color="C2")  #yvalue, xmin, xmax 
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"Error type: {type(e)}")
        return self.create_dict(self.acf_widths)

    def ccf(self):
        #Find the CCF peak at zero time lag and classifies them as bad 
        #CCF Widths = [(Amplitude, Code Index I, Code Index J), ... , ... ]
        #CCF Heights = [(Amplitude, Code Index I, Code Index J), ... , ...]
        #GOOD: CCF WIDTHS MINIMISED
        #GOOD: CCF HEIGHTS MINIMISED 
        # Returns, ccf_widths, pairidx, and       
        ccf_widths = []
        csd_heights = [] #energy at 1/Lc only 
        sum_cfs = [] 
        ccf_width = 0  #if no peak found 
        for i in range(len(self.allcodes)):
            for j in range(i+1, len(self.allcodes)):
                self.ccf = correlate(self.allcodes[i], self.allcodes[j])
                self.fbins, self.csdx = csd(self.allcodes[i], self.allcodes[j], nperseg=self.Lc, return_onesided=False)       
                if not isclose(self.fbins[1], 1/self.Lc): raise ValueError("First frequency bin not close to 1/Lc!!")
                csd_height =  float(self.fbins[1])
                csd_heights.append((f"{i}-{j}", csd_height))  

                peaks, _ = find_peaks(self.ccf)
                try:
                    largest_peak_index = peaks[np.argmax(self.ccf[peaks])]
                    #max width at half max 
                    results_half = peak_widths(self.ccf, [largest_peak_index], rel_height=0.5) 
                    ccf_width = float(results_half[3][0]-results_half[2][0])
                    ccf_widths.append( (f"{i}-{j}", ccf_width) )                    

                    lagx1, lagx2 = self.lagscaler(results_half)
                    plt.plot(self.lags[largest_peak_index] , self.ccf[largest_peak_index ], "x")
                    plt.hlines(results_half[1], lagx1, lagx2, color="C2")  #yvalue, xmin, xmax 
            
                except Exception as e:
                    print(f"Error occurred: {e}")
                    print(f"Error type: {type(e)}")

                sumcf = csd_height + ccf_width 
                sum_cfs.append((f"{i}-{j}", sumcf))

        return self.create_dict(sum_cfs)


    def plot(self, output):
        if output == "ACF":
            for j in range(len(self.allcodes)):
                #plot and save images 
                plt.figure()
                plt.plot(self.lags, acf)
                plt.xlabel("Lag")
                plt.ylabel("ACF")
                plt.grid()
                plt.title(f"ACF: code[{j}]")

                plt.savefig(f"./results/{self.codefam}/acf/acf_code{j}.png")
                plt.close()
        elif output == "CCF":
            for i in range(len(self.allcodes)):
                for j in range(i+1, len(self.allcodes)):
                    #plot and save images 
                    plt.figure()
                    plt.plot(self.lags,self.ccf)
                    plt.grid()
                    plt.title(f"CCF: code[{i}]xcode[{j}]")
                    plt.savefig(f"./results/{self.codefam}/ccf/ccf_codes{i}x{j}.png")
                    plt.close()
        elif output == "CSD":
            for i in range(len(self.allcodes)):
                for j in range(i+1, len(self.allcodes)):
                    plt.figure()
                    plt.stem(self.fbins,np.abs(self.csdx))
                    plt.title(f"CSD: code[{i}]xcode[{j}]")
                    for m in range(-self.Lc//2, self.Lc//2 + 1):
                        plt.axvline(m / self.Lc, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  
                    k_vals = np.arange(-self.Lc//2, self.Lc//2 + 1, 4 )   # step = 4
                    ticks = k_vals / self.Lc
                    labels = [rf"{k}/Lc" for k in k_vals]
                    plt.xticks(ticks, labels)
                    plt.savefig(f"./results/{self.codefam}/csd/csd_codes{i}x{j}.png")
                    plt.close()
        
        
    def lagscaler(self, results_half):

        left_ip = results_half[2][0]
        right_ip = results_half[3][0]

        left_lag = np.interp(left_ip, np.arange(len(self.lags)), self.lags)
        right_lag = np.interp(right_ip, np.arange(len(self.lags)), self.lags)

        return  left_lag, right_lag 


    def sf(self):
        #here i calculate the spectral flatness (sf) of a code
        #GOOD: MAXIMISED
        sfs = []
        for j in range(len(self.allcodes)):
            fftcode = fft(self.allcodes[j])
            power = np.abs(fftcode)**2 + 1e-12 #added 1e-12 to account for 0 bins, but close enough to be zero. 
            normpower = power / np.sum(power)
            amean = np.mean(normpower)
            gmean = np.exp(np.mean(np.log(normpower))) 
            sf = float(gmean / amean)
            sfs.append((sf, j))
        return sfs

    def find_best_clique(self, nodeweights, edgeweights, Narrays):

        best_nodes = None
        best_weight = float('inf')

        for clique in itertools.combinations(range(len(self.allcodes)),Narrays):
            node_sum = sum(nodeweights[i] for i in clique)
            edge_sum =  sum(edgeweights[f"{i}-{j}"] for i, j in itertools.combinations(clique, 2))

            total = node_sum + edge_sum
            if total < best_weight:
                best_weight = total
                best_nodes = clique

        return best_nodes

    def create_dict(self, list):
        #create dict for the nodes and vertices
        #nodes [ key : value ] = [ code index : ACF_width ]
        #edges [ key : value ] = [ code]
        node_weights = {}
        for _, (key, value) in enumerate(list):
            node_weights[key] = value 
        return node_weights             

    

if __name__ == "__main__":

    #rankings of acf widths, ccf widths, and csd height at 1/Lc

    #initiate code config
    WHcfg = dbfCfg(Lc=32, code="WH")
    WHcodes = CodeSelector(WHcfg)
    node_weights = WHcodes.acf()
    edge_weights = WHcodes.ccf()
    best_clique = WHcodes.find_best_clique(node_weights, edge_weights, 4)
    print(best_clique)

    #proof, run BER curves and see if we do get improvements 


    #Goldcfg = dbfCfg(Lc=7, code="Gold")
    #Goldcodes = CodeSelector(Goldcfg)
    #Goldcodes.acf()
    #Goldcodes.ccf()
    #print(Goldcodes.sf())
