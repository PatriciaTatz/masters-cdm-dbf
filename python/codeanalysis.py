"""
analyse code sequences and find optimal set
assuming that its true Lc  = Narrays * 2 
"""

import itertools
from math import isclose

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.signal import (correlate, correlation_lags, csd, find_peaks,
                          peak_widths, peak_prominences)

from components import CDMx
from config import CDMcfg
from models import CodeWeights


class CodeSelector:
    #Method to rank and select codes 
    def __init__(self, conf):
        self.Lc = conf.lc
        self.codefam = conf.codefam
        self.codeids = conf.codeids

        cdm = CDMx(conf)
        self.allcodes = cdm.generate_codes()
        self.lags = correlation_lags(self.Lc, self.Lc)

        self.node_weight = {i: 0 for i in self.codeids}
        self.edge_weight = {
            f"{i}-{j}": 0
            for i, j in itertools.combinations(self.codeids, 2)
        }

    def generate_weights(self):
        self.sfs =  self.sf() #how flat spectrum is 
        self.acfs = self.acf() #width of zero-lag ACF 
        
        ccs = self.ccf()

        self.ccs_lookup = {
            (ci, cj): (ccf_width, ccf_peak, csd_height)
            for ci, cj, ccf_width, ccf_peak, csd_height in ccs
        }

        #for both node and edge, large weight BAD, small weight GOOD, total algo minimises weight 
        self.codeids = [int(x) for x in self.codeids]
        self.node_weight = {
            codeid: float(0.5*acf + 0.5*sf)
            for codeid, sf, acf in zip(self.codeids, self.sfs, self.acfs)
        }

        for ci, cj, ccf_width, ccf_peak, csd_height in ccs:
            """
            self.edge_weight[f"{i}-{j}"] += (
                1/2*csd_height +
                1/2*abs(ccf_peak) 
            )
            """
            self.edge_weight[f"{ci}-{cj}"] += (
                1/3*csd_height+
                1/3*ccf_width+
                1/3*abs(ccf_peak)
            )
            
        """
        plt.figure(figsize=(6,4))
        plt.hist(edge_vals, bins=30)
        plt.xlabel("Edge weight")
        plt.ylabel("Count")
        plt.grid()
        plt.title("Distribution of Edge Weights")

        node_vals = list(self.node_weight.values())

        plt.figure(figsize=(6,4))
        plt.hist(node_vals, bins=20)
        plt.xlabel("Node weight")
        plt.ylabel("Count")
        plt.title("Distribution of Node Weights")
        plt.grid()
        #plt.show()
        """
        return CodeWeights(self.node_weight, self.edge_weight)

    def sf(self):
        """
        Compute the spectral flatness (SF) for each code.
        Spectral flatness measures how noise-like a spectrum is:
            SF = geometric mean / arithmetic mean
        Values near:
            1 -> flat, noise-like spectrum (GOOD)
            0 -> peaky, tonal spectrum (BAD)
        Ranking:
            Higher spectral flatness is better.
        Returns:
            sf_score : ndarray
            Normalized spectral flatness score for each code.
        """

        sf_score = []
        self.sf_plot = []

        for code in self.allcodes:
            # Power spectrum
            Xi = np.fft.fft(code)
            power = np.real((Xi*np.conj(Xi))/ np.abs(len(code))**2)
            power += 1e-12 #handles zero within np.log

            # Arithmetic and geometric means
            arithmetic_mean = np.mean(power)
            geometric_mean = np.exp(np.mean(np.log(power)))
            # Spectral flatness
            flatness = geometric_mean / arithmetic_mean
            sf_score.append(flatness)
            self.sf_plot.append(power)

        # Normalize scores to [0, 1] and handles very small sf values
        sf_score = -np.log10(sf_score) 
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
        Values near:
            1 -> Has longer decay at zero lag thus BAD (heavy weight)
            0 -> Minimal decay at zero lag thus GOOD (light weight)
        Returns:
            acf_score : ndarray
        """

        self.acfplot = []
        acf_score = []
        for code in self.allcodes:
            acf = correlate(code, code, mode='full')

            zero_lag = len(acf) // 2
            prominence = peak_prominences(acf, [zero_lag])[0][0]
            if prominence < 1e-6:
                # No actual peak
                widths = [self.Lc]    # no peak will result in heavy weight
                width_heights = 0
                left_ips = 0 
                right_ips = 0 
            else:
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
        acf_score = np.asarray(acf_score) / np.max(acf_score)
        return acf_score 
        
    def ccf(self):
        """
        Returns
        -------
        metrics : list of tuples
            (i, j, ccf_width, ccf_peak, csd_height)
        """

        self.ccf_data = []
        self.csd_data = []
        metrics = []

        for i in range(len(self.allcodes)):
            for j in range(i + 1, len(self.allcodes)):
                ccf = correlate(self.allcodes[i], self.allcodes[j], mode='full') #normalise ccf as peak height is considered

                zero_lag = len(ccf)//2
                prominence = peak_prominences(ccf, [zero_lag])[0][0]
                if ccf[zero_lag] == 0:
                    # No actual peak
                    widths = [0]   # no peak result in lightest weight
                    heights = [0]
                    left_ips = [0] 
                    right_ips = [0] 
                else: 
                    widths, heights, left_ips, right_ips = peak_widths(
                        ccf,
                        [zero_lag],
                        rel_height=0.5
                    )

                self.ccf_data.append(
                    (
                        self.codeids[i],
                        self.codeids[j],
                        ccf,
                        float(widths[0]),
                        float(heights[0]),
                        float(left_ips[0]),
                        float(right_ips[0]),
                    )
                )
                """
                # CSD
                fbins,csdx  = csd(
                    self.allcodes[i],
                    self.allcodes[j],
                    nperseg=self.Lc,
                    return_onesided=False
                )
                """
                codes_mixed = self.allcodes[i]*self.allcodes[j]
                Sxy = np.fft.fft(codes_mixed)
                fbins = np.fft.fftfreq(self.Lc, d=1)

                self.csd_data.append(
                    (
                        self.codeids[i],
                        self.codeids[j],
                        fbins,
                        Sxy,
                    )
                )

                """
                #sanitecheck
                if self.codeids[i]==2 and self.codeids[j]==3:
                    print(fbins)
                    print(np.abs(csdx[1]))
                
                if not isclose(fbins[1], 1 / self.Lc):
                    raise ValueError(
                        f"Expected first frequency bin = {1/self.Lc}, got {fbins[1]}"
                    )
                """
                metrics.append(
                    (
                        int(self.codeids[i]),
                        int(self.codeids[j]),
                        float(widths[0]),      # CCF width
                        float(ccf[zero_lag]),   # zero-lag CCF peak as plotted
                        float(np.abs(Sxy[1])) # CSD amplitude at 1/Lc
                    )
                )

        width_max = max(m[2] for m in metrics)
        peak_max  = max(abs(m[3]) for m in metrics)
        csd_max   = max(m[4] for m in metrics)

        metrics_norm = [
            (
                ci,
                cj,
                width / width_max if width_max else 0.0,
                abs(peak) / peak_max if peak_max else 0.0,
                csd / csd_max if csd_max else 0.0,
            )
            for ci, cj, width, peak, csd in metrics
        ]
        return metrics_norm
        
    def find_best_clique(self, weights, no_elem):
        cliques = []
        totals = []
        for clique in itertools.combinations(self.codeids, no_elem):            
            node_mean =sum(weights.node[int(i)] for i in clique)/no_elem
            edge_mean = np.mean([
                weights.edge[f"{i}-{j}"]
                for i, j in itertools.combinations(clique, 2)
            ])
            total = float(edge_mean + node_mean)
            cliques.append(clique)
            totals.append(total)

            """
            if clique == (2, 3):
                print("(2,3)", node_mean, edge_mean, total)
            if clique == (15, 16):
                print("(15,16)", node_mean, edge_mean, total)                
            """

        result = dict(zip(cliques,totals))
        self.best = min(result, key=result.get)
        self.worst = max(result, key=result.get)
        """
        n = 10
        for key, value in sorted(result.items(), key=lambda x: x[1], reverse=False)[:n]:
            print(f"{key}: {value}")
        """
        return self.best, self.worst 
          

    def plot(self, output, cliquetype="best"):
        if output == "SF":
            
            for j in range(len(self.allcodes)):
                plt.rcParams['font.size'] = 15
                plt.figure()
                plt.plot(self.sf_plot[j])
                plt.title(f"Normalised PS Flatnesss of c={self.codeids[j]}")
                plt.xlabel("Frequency bin")
                plt.ylabel("Norm Power")
                plt.grid()
                plt.text(
                    0.02, 0.95,
                    f"Est. Score={self.sfs[j]:.2f}",
                    transform=plt.gca().transAxes,
                    ha='left',
                    va='top',
                    bbox=dict(facecolor='white', edgecolor='black')
                )
                plt.savefig(f"./results/{self.codefam}/sf/sf_code{self.codeids[j]}.png")
                plt.close()
        if output == "ACF":
            for j in range(len(self.allcodes)):
                plt.rcParams['font.size'] = 15
                acf = self.acfplot[j][0]
                height = self.acfplot[j][2]
                lips = np.interp(self.acfplot[j][3], np.arange(len(acf)), self.lags) 
                rips = np.interp(self.acfplot[j][4], np.arange(len(acf)), self.lags) 

                #plot and save images 
                plt.figure()
                plt.plot(self.lags, acf)
                plt.xlabel("Lag")
                plt.ylabel("ACF")
                plt.title(f"ACF: code[{self.codeids[j]}]")
                zerolag = len(acf)//2
                plt.plot(self.lags[zerolag], acf[zerolag], "x")
                plt.hlines(height, lips, rips, color="C2")  #yvalue, xmin, xmax 

                plt.text(
                    0.02, 0.95,
                    f"Score={self.acfs[j]:.2f}",
                    transform=plt.gca().transAxes,
                    ha='left',
                    va='top',
                    bbox=dict(facecolor='white', edgecolor='black')
                )

                plt.grid()
                plt.savefig(f"./results/{self.codefam}/acf/acf_code{self.codeids[j]}.png")
                plt.close()
                
        elif output == "CCF":
            for ci, cj, ccf, width, height, left_ip, right_ip in self.ccf_data:
                left_lag = np.interp(
                    left_ip,
                    np.arange(len(ccf)),
                    self.lags
                )

                right_lag = np.interp(
                    right_ip,
                    np.arange(len(ccf)),
                    self.lags
                )
                plt.rcParams['font.size'] = 15

                plt.figure()
                plt.plot(self.lags, ccf)
                # Zero-lag peak
                zerolag = len(ccf)//2
                plt.plot(
                    self.lags[zerolag],
                    ccf[zerolag],
                    "x",
                    color="C1"
                )
                # Peak width
                plt.hlines(
                    height,
                    left_lag,
                    right_lag,
                    color="C2",
                    linewidth=2
                )
                plt.ylabel("CCF")
                plt.xlabel("Lag")

                plt.grid(True)
                plt.title(f"CCF: code[{ci}] x code[{cj}]")

                ccf_width, ccf_peak, _ = self.ccs_lookup[(ci, cj)]
                plt.text(
                    0.02, 0.95,
                    (
                        f"CCF width = {ccf_width:.3f}\n"
                        f"CCF peak = {ccf_peak:.3f}"
                    ),
                    transform=plt.gca().transAxes,
                    ha='left',
                    va='top',
                    bbox=dict(facecolor='white', edgecolor='black')
                )

                plt.savefig(
                    f"./results/{self.codefam}/ccf/ccf_codes{ci}x{cj}.png"
                )

                plt.close()

        elif output == "CSD":
            for ci, cj, fbins, csd in self.csd_data:
                plt.rcParams['font.size'] = 15

                plt.figure()
                plt.stem(fbins, np.abs(csd))
                plt.title(f"CSD: code[{ci}] x code[{cj}]")

                for m in range(-self.Lc//2, self.Lc//2 + 1):
                    plt.axvline(
                        m / self.Lc,
                        color="gray",
                        linestyle="--",
                        linewidth=0.5,
                        alpha=0.5
                    )

                k_vals = np.arange(-self.Lc//2, self.Lc//2 + 1, 4)
                plt.xticks(
                    k_vals / self.Lc,
                    [rf"{k}/Lc" for k in k_vals]
                )

                _ , _ , csd_height = self.ccs_lookup[(ci, cj)]
                plt.text(
                    0.02, 0.95,
                    (
                        f"CSD height = {csd_height:.3f}"
                    ),
                    transform=plt.gca().transAxes,
                    ha='left',
                    va='top',
                    bbox=dict(facecolor='white', edgecolor='black')
                )

                plt.savefig(
                    f"./results/{self.codefam}/csd/csd_codes{ci}x{cj}.png"
                )
                plt.close()

        if output == "network":
            if cliquetype == "best":
                clique = self.best
            elif cliquetype == "worst":
                clique = self.worst

            G = nx.Graph()
            
            G.add_nodes_from(clique)
            G.add_edges_from(itertools.combinations(clique, 2))
            pos = nx.spring_layout(G)

            values = [self.node_weight[n] for n in G.nodes()]

            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=[1000 + v*100  for v in values],
                node_color=values,
                cmap=plt.cm.coolwarm,
            )

            nx.draw_networkx_edges(
                G,
                pos,
                width=[
                    0.3+2*self.edge_weight.get(f"{i}-{j}", self.edge_weight.get(f"{j}-{i}", 1))
                    for i, j in G.edges()
                ]            )

            nx.draw_networkx_labels(G, pos)
            plt.rcParams['font.size'] = 20

            plt.title("All Possible Code Pairs")
            plt.show()


if __name__ == "__main__":
    #initiate code config
    lc = 32
    ids = np.arange(1,lc)
    WHcfg = CDMcfg(
        simtime = 1, 
        lc = lc,
        codefam="WH",
        no_elem= 6,
        codeids=ids)


    WHcodes = CodeSelector(WHcfg)
    weights = WHcodes.generate_weights()
    best, worst = WHcodes.find_best_clique(weights, WHcfg.no_elem)
    print("best =",best)
    print("worst = ", worst)
    #WHcodes.plot("CSD")
    #WHcodes.plot("ACF")
    #WHcodes.plot("CCF")
    #WHcodes.plot("SF")
    #WHcodes.plot("network", "best")
