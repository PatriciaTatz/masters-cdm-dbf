"""
analyse code sequences and find optimal set
assuming that its true Lc  = Narrays * 2 
"""

from components import CodeGenerator
from config import dbfCfg, signalCfg

from scipy.signal import find_peaks, peak_widths, correlate, correlation_lags, csd
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.fft import fft, fftfreq
import itertools
from math import isclose
import networkx as nx


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
        acf_widths = []
        self.acf_plot = []
        for j in range(len(self.allcodes)):
            print(self.allcodes[j])
            #periodic ACF and normalised 
            acf = np.real(np.fft.ifft(np.fft.fft(self.allcodes[j]) * np.conj(np.fft.fft(self.allcodes[j]))))
            acf = np.fft.fftshift(acf) / self.Lc 
            try:
                zerolagindex = self.Lc //2 
                acf_pk = peak_widths(acf, [self.Lc//2], rel_height=0.5) 
                print(acf_pk)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"Error type: {type(e)}")
        

            self.acf_plot.append((acf,) + acf_pk)
            acf_widths.append((j, acf_pk[0]))

        acf_widths = self.create_dict(acf_widths)

        return acf_widths

    def ccf(self):
        #Find the CCF peak at zero time lag and classifies them as bad 
        #CCF Widths = [(Amplitude, Code Index I, Code Index J), ... , ... ]
        #CCF Heights = [(Amplitude, Code Index I, Code Index J), ... , ...]
        #GOOD: CCF WIDTHS MINIMISED
        #GOOD: CCF HEIGHTS MINIMISED 
        # Returns, ccf_widths, pairidx, and       

        ccf_width = 0  #if no peak found 

        self.csd_plot = []
        self.ccf_plot = []
        
        sum_cfs = [] 
        for i in range(len(self.allcodes)):
            for j in range(i+1, len(self.allcodes)):
                #PERIODIC CCF 
                ccf = np.fft.ifft(np.fft.fft(self.allcodes[i]) * np.conj(np.fft.fft(self.allcodes[j])))
                ccf = np.fft.fftshift(ccf) 
                try:
                    zerolagindex = self.Lc //2 
                    ccf_width , _, left, right = peak_widths(ccf, [self.Lc//2], rel_height=0.5) 
                    
                except Exception as e:
                        print(f"Error occurred: {e}")
                        print(f"Error type: {type(e)}")
                ccf_plot.append((f"{i}-{j}", [ccf_width, left, right]))   

                #CSD               
                self.fbins, csdx = csd(self.allcodes[i], self.allcodes[j], nperseg=self.Lc, return_onesided=False)  
                self.csdx_list.append(csdx)     
                if not isclose(self.fbins[1], 1/self.Lc): raise ValueError("First frequency bin not close to 1/Lc!!")
                csd_height =  float(self.fbins[1])
                csd_heights.append((f"{i}-{j}", csd_height))  

                sumcf = csd_height + ccf_width 
                sum_cfs.append((f"{i}-{j}", sumcf))
        sum_cfs = self.create_dict(sum_cfs)

        return sum_cfs

    def sf(self):
        #here i calculate the spectral flatness (sf) of a code
        #GOOD: MAXIMISED
        self.sfs = []
        for j in range(len(self.allcodes)):
            fftcode = fft(self.allcodes[j])
            power = np.abs(fftcode)**2 + 1e-12 #added 1e-12 to account for 0 bins, but close enough to be zero. 
            normpower = power / np.sum(power)
            amean = np.mean(normpower)
            gmean = np.exp(np.mean(np.log(normpower))) 
            sf = float(gmean / amean)
            self.sfs.append((j, sf))
        
        self.sfs = self.create_dict(self.sfs)

        return self.sfs
        
    def lagscaler(self, results_half):

        left_ip = results_half[2][0]
        right_ip = results_half[3][0]

        left_lag = np.interp(left_ip, np.arange(len(self.lags)), self.lags)
        right_lag = np.interp(right_ip, np.arange(len(self.lags)), self.lags)

        return  left_lag, right_lag 


    def find_best_clique(self, nodeweights, edgeweights, Narrays):

        self.best_nodes = None
        self.worst_nodes = None 
        best_weight = float('inf')
        worst_weight = float('-inf')

        for clique in itertools.combinations(range(len(self.allcodes)),Narrays):
            node_sum = sum(nodeweights[i] for i in clique)
            edge_sum =  sum(edgeweights[f"{i}-{j}"] for i, j in itertools.combinations(clique, 2))

            total = node_sum + edge_sum
            if total < best_weight:
                best_weight = total
                self.best_nodes = clique
            if total > worst_weight:
                worst_weight = total
                self.worst_nodes = clique

        return self.best_nodes, self.worst_nodes

    def create_dict(self, list):
        #create dict for the nodes and vertices
        #nodes [ key : value ] = [ code index : ACF_width ]
        #edges [ key : value ] = [ code]
        node_weights = {}
        for _, (key, value) in enumerate(list):
            node_weights[key] = value 
        return node_weights             

    def plot(self, output):
        if output == "ACF":
            for j in range(len(self.allcodes)):

                lags = np.arange(-self.Lc//2, self.Lc//2)
                acf = self.acf_plot[j][0]
                height = self.acf_plot[j][2]
                lips = np.interp(self.acf_plot[j][3], np.arange(self.Lc), lags) 
                rips = np.interp(self.acf_plot[j][4], np.arange(self.Lc), lags) 

                #plot and save images 
                plt.figure()
                plt.plot(lags, acf)
                plt.xlabel("Lag")
                plt.ylabel("ACF")
                plt.grid()
                plt.title(f"ACF: code[{j}]")

                plt.plot(lags[self.Lc//2], acf[self.Lc//2], "x")
                plt.hlines(height, lips, rips, color="C2")  #yvalue, xmin, xmax 
                plt.savefig(f"./results/{self.codefam}/acf/acf_code{j}.png")
                plt.close()
        elif output == "CCF":
            for i in range(len(self.allcodes)):
                for j in range(i+1, len(self.allcodes)):
                    #plot and save images 



                    plt.figure()
                    plt.plot(self.lags,self.ccf)
                    lagx1, lagx2 = self.lagscaler(self.ccf_results_half)
                    plt.plot(self.lags[self.ccf_largest_peak_index] , self.ccf[self.ccf_largest_peak_index ], "x")
                    plt.hlines(self.ccf_results_half[1], lagx1, lagx2, color="C2")  #yvalue, xmin, xmax 
                    plt.grid()
                    plt.title(f"CCF: code[{i}]xcode[{j}]")
                    plt.savefig(f"./results/{self.codefam}/ccf/ccf_codes{i}x{j}.png")
                    plt.close()



        elif output == "CSD":
            for i in range(len(self.allcodes)):
                for j in range(i+1, len(self.allcodes)):
                    plt.figure()
                    plt.stem(self.fbins,np.abs(self.csdx_list[i+j]))
                    plt.title(f"CSD: code[{i}]xcode[{j}]")
                    for m in range(-self.Lc//2, self.Lc//2 + 1):
                        plt.axvline(m / self.Lc, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  
                    k_vals = np.arange(-self.Lc//2, self.Lc//2 + 1, 4 )   # step = 4
                    ticks = k_vals / self.Lc
                    labels = [rf"{k}/Lc" for k in k_vals]
                    plt.xticks(ticks, labels)
                    plt.savefig(f"./results/{self.codefam}/csd/csd_codes{i}x{j}.png")
                    plt.close()

        if output == "network":
            clique = self.worst_nodes
            G = nx.Graph()
            G.add_nodes_from(clique)
            G.add_edges_from(itertools.combinations(clique, 2))
            pos = nx.spring_layout(G)

            node_weights = [self.acf_widths[i] + self.sfs[i] for i in G.nodes()]
            #"1-3" and "3-1" are different keys so thats why it tries both below
            edge_weights = [self.sum_cfs.get(f"{i}-{j}", self.sum_cfs.get(f"{j}-{i}", 1)) for i, j in G.edges()]
            plt.figure()
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=[w * 900 for w in node_weights],
                node_color=node_weights,
                cmap=plt.cm.coolwarm
            )
            nx.draw_networkx_edges(
                G,
                pos,
                width=edge_weights
            )
            labels = {
                i: f"c{i}\n{self.acf_widths[i] + self.sfs[i]:.2f}"
                for i in G.nodes()
            }
            nx.draw_networkx_labels(G, pos, labels=labels)
            edge_labels = {
                (i, j): round(self.sum_cfs.get(f"{i}-{j}", self.sum_cfs.get(f"{j}-{i}", 1)),4)
                for i, j in G.edges()
            }
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                label_pos=0.3,   # move closer to source (0 → node i, 1 → node j)
                font_size=8
            )
            plt.show()  


if __name__ == "__main__":

    #rankings of acf widths, ccf widths, and csd height at 1/Lc

    #initiate code config
    WHcfg = dbfCfg(Lc=16, code="WH", no_arrays=2)
    WHcodes = CodeSelector(WHcfg)
    acfs = WHcodes.acf()
    #ccfs = WHcodes.ccf()
    #sfs = WHcodes.sf()

    #node_weights = {k: acfs[k] + sfs[k] for k in acfs.keys() & sfs.keys()}
    #edge_weights = ccfs
    #best, worst = WHcodes.find_best_clique(node_weights, edge_weights, WHcfg.no_arrays)
    WHcodes.plot("ACF")

    #WHcodes.plot("network")

    #WHcfg.codeids = best 
    #print(worst)
