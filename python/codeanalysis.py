"""
analyse code sequences and find optimal set
assuming that its true Lc  = Narrays * 2 
"""

from components import CDMx
from config import CDMcfg   

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
        self.disqualified = []


    def acf(self):
        #Find the peak width at zero timelag
        #acf = [acf, pk width, pk width_height, pk left ips, pk right ips ]
        #Rank: ACF WIDTHS MINIMISED 
        acf_score = []
        self.acf = []
        for j in range(len(self.allcodes)):
            #periodic ACF and normalised 
            acf = np.real(np.fft.ifft(np.fft.fft(self.allcodes[j]) * np.conj(np.fft.fft(self.allcodes[j]))))
            acf = np.fft.fftshift(acf) / self.Lc 
            
            zerolagindex = self.Lc //2 
            acf_pk = peak_widths(acf, [self.Lc//2], rel_height=0.5)

            #force correct flat acf as BAD scores 
            if acf_pk[0] == 0:
                del self.node_weight[j]
                self.disqualified.append(j)

            self.acf.append((acf,) + acf_pk)
            acf_score.append(acf_pk[0][0] + acf_pk[1][0])

        acf_score = acf_score / np.max(acf_score) #normalised
        self.node_weight = {
            i: self.node_weight[i] / acf_score[i]
            for i in range(len(acf_score))
            if i not in self.disqualified
            }

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
                
                self.edge_weight[f"{i}-{j}"] += csd_height + ccf_pk[0][0] + ccf_pk[1][0]


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
            sfs.append(sf)
        sfs = sfs / np.max(sfs)

        self.node_weight = {
            i: sfs[i]
            for i in range(len(sfs))
            if i not in self.disqualified
        }

        

    def find_best_clique(self, Narrays):
        self.best_nodes = None
        self.worst_nodes = None 
        best_weight = float('inf')
        worst_weight = float('-inf')

        #this order of operation is important! 
        WHcodes.sf()
        WHcodes.acf()
        WHcodes.ccf()

        #normalise the self node weights and edge weights 
        self.node_weight = {
            k: v / sum(self.node_weight.values())
            for k, v in self.node_weight.items()
        }


        valid_codes = [i for i in range(self.Lc) if i not in self.disqualified]
        if len(valid_codes)<Narrays: raise("Not enough valid codes for this N array size!")

        for clique in itertools.combinations(valid_codes,Narrays):
            print(self.node_weight)
            node_sum = sum(self.node_weight[i] for i in clique)
            edge_sum =  sum(self.edge_weight[f"{i}-{j}"] for i, j in itertools.combinations(clique, 2))

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


    #rankings of acf widths, ccf widths, and csd height at 1/Lc

    #initiate code config
    WHcfg = CDMcfg(
        simtime = 1, 
        lc=8,
        codefam="WH",
        no_elem=7)
    WHcodes = CodeSelector(WHcfg)
    best, worst = WHcodes.find_best_clique(WHcfg.no_elem)
    WHcodes.plot("ACF")
    WHcodes.plot("CCF")
    WHcodes.plot("CSD")

    WHcodes.plot("network", "best")

    #WHcfg.codeids = best 
    #print(worst)
