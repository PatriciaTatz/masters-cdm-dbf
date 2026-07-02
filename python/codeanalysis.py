"""
analyse code sequences and find optimal set
assuming that its true Lc  = Narrays * 2 
"""

from components import CDMx
from config import CDMcfg   
from models import CodeWeights

from scipy.signal import find_peaks, peak_widths, correlate, correlation_lags, csd
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors 
import numpy as np 
from scipy.fft import fft, fftfreq
import itertools
from math import isclose
import networkx as nx


class CodeSelector:
    #Method to rank and select codes 
    def __init__(self, conf):
        self.Lc = conf.lc
        self.codefam = conf.codefam

        cdm = CDMx(conf)
        self.allcodes = cdm.generate_codes()
        self.lags = correlation_lags(self.Lc, self.Lc)
        self.node_weight = {i: 0 for i in range(self.Lc)}
        self.edge_weight = {f"{i}-{j}":0 for i,j in itertools.combinations(range(self.Lc), 2)}

    def generate_weights(self):
        sfs =  self.sf()
        acfs = self.acf()
        
        self.node_weight = {
            i: 0.5*acfs[i] + 0.5*sfs[i]
            for i in range(self.Lc)
            }

        csd_height, ccf_pk = self.ccf()
        self.edge_weight[f"{i}-{j}"] += csd_height + ccf_pk[0][0] + ccf_pk[1][0]

        return CodeWeights(node, edges)

    def sf(self):
        """
        Compute the spectral flatness (SF) for each code.

        Spectral flatness measures how noise-like a spectrum is:

            SF = geometric mean / arithmetic mean

        Values near:
            1 -> flat, noise-like spectrum
            0 -> peaky, tonal spectrum

        Ranking:
            Higher spectral flatness is better.

        Returns:
            sf_score : ndarray
                Normalized spectral flatness score for each code.
        """

        sf_score = []

        for code in self.allcodes:

            # Power spectrum
            spectrum = fft(code)
            power = np.abs(spectrum) ** 2

            # Prevent log(0) in geometric mean calculation
            power += 1e-12

            # Normalize power spectrum
            norm_power = power / np.sum(power)

            # Arithmetic and geometric means
            arithmetic_mean = np.mean(norm_power)
            geometric_mean = np.exp(np.mean(np.log(norm_power)))

            # Spectral flatness
            flatness = geometric_mean / arithmetic_mean

            sf_score.append(flatness)

        # Normalize scores to [0, 1]
        sf_score = np.asarray(sf_score)
        sf_score /= np.max(sf_score)

        return sf_score


    def acf(self):
        """
        Compute the periodic autocorrelation function (ACF) for each code and
        measure the width of the zero-lag peak.

        Stores:
            self.acf_data : list of tuples
                (acf, widths, width_heights, left_ips, right_ips)
                Used for plotting 

        Returns:
            acf_score : normalized peak-width score
        """

        self.acfplot = []
        acf_score = []

        zero_lag = self.Lc // 2

        for code in self.allcodes:
            # Periodic autocorrelation (Wiener-Khinchin theorem)
            acf = np.real(
                np.fft.ifft(
                    np.fft.fft(code) *
                    np.conj(np.fft.fft(code))
                )
            )

            # Shift zero lag to centre and normalise
            acf = np.fft.fftshift(acf) / self.Lc

            # Measure full width at half maximum (FWHM)
            widths, width_heights, left_ips, right_ips = peak_widths(
                acf,
                [zero_lag],
                rel_height=0.5
            )

            # Store full result for later inspection
            self.acfplot.append(
                (acf, widths, width_heights, left_ips, right_ips)
            )

            # Narrower peaks are better
            acf_score.append(widths[0])

        # Normalize scores to [0, 1]
        acf_score = np.asarray(acf_score)
        acf_score /= np.max(acf_score)

        return acf_score 
        

    def ccf(self):
        #Find the CCF peak at zero time lag and classifies them as bad 
        #CCF Widths = [(Amplitude, Code Index I, Code Index J), ... , ... ]
        #CCF Heights = [(Amplitude, Code Index I, Code Index J), ... , ...]
        #GOOD: CCF WIDTHS MINIMISED
        #GOOD: CCF HEIGHTS MINIMISED 
        # Returns, ccf_widths, pairi, and       

        self.csd = []
        self.ccf = []
        for i in range(len(self.allcodes)):
            for j in range(i+1, len(self.allcodes)):
                #PERIODIC CCF and normalised
                #ccf_peak = [i, j, ccf, pk width, pk width_height, pk left ips, pk right ips ]
                ccf = np.real(np.fft.ifft(np.fft.fft(self.allcodes[i]) * np.conj(np.fft.fft(self.allcodes[j]))))
                ccf = np.fft.fftshift(ccf) / self.Lc 

                zerolagindex = self.Lc //2 
                ccf_pk = peak_widths(ccf, [self.Lc//2], rel_height=0.5) 
                self.ccf.append((i, j, ccf, ) + ccf_pk)   
                
                #CSD               
                self.fbins, csdx = csd(self.allcodes[i], self.allcodes[j], nperseg=self.Lc, return_onesided=False)  
                self.csd.append((i,j, csdx))  

                #Amplitude at 1/Lc 
                if not isclose(self.fbins[1], 1/self.Lc): raise ValueError("First frequency bin not close to 1/Lc!!")
                csd_height =  float(self.fbins[1])

        return csd_heights, ccf_pk

        
    def find_best_clique(self, weights, no_elem):
        self.best_nodes = None
        self.worst_nodes = None 
        best_weight = float('inf')
        worst_weight = float('-inf')

        valid_codes = [i for i in range(self.Lc) if i not in self.disqualified]
        print(valid_codes)
        if len(valid_codes)<no_elem: raise("Not enough valid codes for this N array size!")

        for clique in itertools.combinations(valid_codes,no_elem):
            node_sum = sum(weights.node[i] for i in clique)
            edge_sum =  sum(weights.edge[f"{i}-{j}"] for i, j in itertools.combinations(clique, 2))

            total = node_sum + edge_sum
            if total < best_weight:
                best_weight = total
                self.best_nodes = clique
            if total > worst_weight:
                worst_weight = total
                self.worst_nodes = clique

        return self.best_nodes, self.worst_nodes
          

    def plot(self, output, cliquetype="best"):
        if output == "ACF":
            for j in range(len(self.allcodes)):
                lags = np.arange(-self.Lc//2, self.Lc//2)
                acf = self.acf[j][0]
                height = self.acf[j][2]
                lips = np.interp(self.acf[j][3], np.arange(self.Lc), lags) 
                rips = np.interp(self.acf[j][4], np.arange(self.Lc), lags) 

                #plot and save images 
                plt.figure()
                plt.plot(lags, acf)
                plt.xlabel("Lag")
                plt.ylabel("ACF")
                plt.grid()
                plt.title(f"ACF: code[{j}]")

                plt.plot(lags[self.Lc//2], acf[self.Lc//2], "x")
                plt.hlines(height, lips, rips, color="C2")  #yvalue, xmin, xmax 
                plt.grid()
                plt.savefig(f"./results/{self.codefam}/acf/acf_code{j}.png")
                plt.close()
        elif output == "CCF":
            for i in range(len(self.ccf)):
                #plot and save images 
                lags = np.arange(-self.Lc//2, self.Lc//2)
                ccf = self.ccf[i][2]
                height = self.ccf[i][4]
                lips = np.interp(self.ccf[i][5], np.arange(self.Lc), lags) 
                rips = np.interp(self.ccf[i][6], np.arange(self.Lc), lags) 
                ci = self.ccf[i][0]
                cj = self.ccf[i][1]

                plt.figure()
                plt.plot(lags,ccf)
                plt.plot(lags[self.Lc//2], ccf[self.Lc//2], "x")
                plt.hlines(height, lips, rips, color="C2")  #yvalue, xmin, xmax 
                plt.grid()
                plt.title(f"CCF: code[{ci}]xcode[{cj}]")
                plt.savefig(f"./results/{self.codefam}/ccf/ccf_codes{ci}x{cj}.png")
                plt.close()
        elif output == "CSD":
            for i in range(len(self.csd)):
                ci = self.csd[i][0]
                cj = self.csd[i][1]
                csd = self.csd[i][2]

                plt.figure()
                plt.stem(self.fbins,np.abs(csd))
                plt.title(f"CSD: code[{ci}]xcode[{cj}]")
                for m in range(-self.Lc//2, self.Lc//2 + 1):
                    plt.axvline(m / self.Lc, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  
                k_vals = np.arange(-self.Lc//2, self.Lc//2 + 1, 4 )   # step = 4
                ticks = k_vals / self.Lc
                labels = [rf"{k}/Lc" for k in k_vals]
                plt.xticks(ticks, labels)
                plt.savefig(f"./results/{self.codefam}/csd/csd_codes{ci}x{cj}.png")
                plt.close()

        if output == "network":
            if cliquetype == "best":
                clique = self.best_nodes
            elif cliquetype == "worst":
                clique = self.worst_nodes

            G = nx.Graph()
            G.add_nodes_from(clique)
            G.add_edges_from(itertools.combinations(clique, 2))
            pos = nx.circular_layout(G)

            values = [self.node_weight[n] for n in G.nodes()]

            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=[v * 1000 for v in values],
                node_color=values,
                cmap=plt.cm.coolwarm,
            )

            nx.draw_networkx_edges(
                G,
                pos,
                width=[
                    5*self.edge_weight.get(f"{i}-{j}", self.edge_weight.get(f"{j}-{i}", 1))
                    for i, j in G.edges()
                ]
            )

            nx.draw_networkx_labels(G, pos)
            plt.show()


if __name__ == "__main__":
    #initiate code config
    WHcfg = CDMcfg(
        simtime = 1, 
        lc=8,
        codefam="WH",
        no_elem=7)


    WHcodes = CodeSelector(WHcfg)

    weights = WHcodes.generate_weights()
    best, worst = WHcodes.find_best_clique(weights, WHcfg.no_elem)

    WHcodes.plot("ACF")
    WHcodes.plot("CCF")
    WHcodes.plot("CSD")

    WHcodes.plot("network", "best")
