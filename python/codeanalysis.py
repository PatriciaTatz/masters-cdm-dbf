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
from scipy.signal import (correlate, correlation_lags, find_peaks,
                          peak_widths, peak_prominences)
import cProfile
import pstats


from components import CDMx
from config import CDMcfg
from models import CodeWeights


class CodeSelector:
    #Method to rank and select codes 
    def __init__(self, conf):
        self.Lc = conf.lc
        self.codefam = conf.codefam
        self.codeids = conf.codeids
        self.arch = conf.SPREADING_STAGE

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
            codeid: float(0*acf + 1*sf)
            for codeid, sf, acf in zip(self.codeids, self.sfs, self.acfs)
        }

        for ci, cj, ccf_width, ccf_peak, csd_height in ccs:
  
            self.edge_weight[f"{ci}-{cj}"] += (
                1/2*csd_height+
                0*ccf_width+
                1/2*abs(ccf_peak)
            )

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
            acf = np.correlate(code, code, mode='full')

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
                ccf = np.correlate(self.allcodes[i], self.allcodes[j], mode='full') #normalise ccf as peak height is considered
                zero_lag = len(ccf)//2
                prominence = peak_prominences(ccf, [zero_lag])[0][0]

                if self.codefam == "ZC":
                    ccf_at_zero = np.abs(np.real(ccf[zero_lag]))
                    """
                    print(
                        "i=",self.codeids[i],
                        "j=",self.codeids[j],
                        np.real(ccf[zero_lag]),
                        np.abs(ccf[zero_lag])/self.Lc
                        )
                    """
                else:
                    ccf_at_zero = np.abs(ccf[zero_lag])

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
                

                #CSD
                codes_mixed = self.allcodes[i]*np.conj(self.allcodes[j])
                Sxy = np.fft.fft(codes_mixed)

                total_energy = np.sum(np.abs(Sxy[0:self.Lc//2+1])**2)
                if self.arch == "BB":
                    #do not want spectral tone at m = 1
                    
                    lower_half_energy =  np.sum(np.abs(Sxy[1])**2)
                elif self.arch == "IF":
                    #do not want at m=2 
                    lower_half_energy =  np.sum(np.abs(Sxy[2])**2)
                if lower_half_energy == 0:
                    csd = 0 
                else:
                    csd = lower_half_energy/total_energy
                fbins = np.fft.fftfreq(self.Lc, d=1)
                self.csd_data.append(
                    (
                        self.codeids[i],
                        self.codeids[j],
                        fbins,
                        Sxy,
                    )
                )
                
                metrics.append(
                    (
                        int(self.codeids[i]),
                        int(self.codeids[j]),
                        float(widths[0]),       # CCF width
                        float(ccf_at_zero),   # zero-lag CCF peak as plotted
                        float(csd) # CSD amplitude at 1/Lc
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


    def find_best_clique_greedy(self, weights, no_elem):

        # Start with the code having the lowest node weight
        first = min(
            self.codeids,
            key=lambda i: weights.node[int(i)]
        )

        clique = [int(first)]

        node_sum = weights.node[int(first)]
        edge_sum = 0.0

        # Keep adding codes until required clique size is reached
        while len(clique) < no_elem:

            best_code = None
            best_total = np.inf

            # Try each remaining code
            for code in self.codeids:

                code = int(code)

                if code in clique:
                    continue

                # Add candidate node weight
                new_node_sum = node_sum + weights.node[code]

                # Add candidate edge weights
                new_edge_sum = edge_sum

                for existing_code in clique:
                    i, j = sorted((existing_code, code))
                    new_edge_sum += weights.edge[f"{i}-{j}"]

                # Calculate score if this code was added
                new_size = len(clique) + 1
                no_edges = new_size * (new_size - 1) / 2

                node_mean = new_node_sum / new_size
                edge_mean = new_edge_sum / no_edges

                total = float(node_mean + edge_mean)

                # Keep the best candidate
                if total < best_total:
                    best_total = total
                    best_code = code

            # Add the chosen code permanently
            for existing_code in clique:
                i, j = sorted((existing_code, best_code))
                edge_sum += weights.edge[f"{i}-{j}"]

            node_sum += weights.node[best_code]

            clique.append(best_code)

        # Final clique score
        no_edges = no_elem * (no_elem - 1) / 2

        self.best_total = float(
            node_sum / no_elem
            + edge_sum / no_edges
        )

        self.best = clique

        return self.best

    def find_best_clique(self, weights, no_elem):


        self.best = None
        self.worst = None
        self.best_total = np.inf
        self.worst_total = -np.inf 
        self.ranked = []

        avg_node = np.mean(list(weights.node.values()))
        avg_edge = np.mean(list(weights.edge.values()))
        threshold = None # 0.75*(avg_node + avg_edge)
        self.dfs_recursive(
            weights,
            no_elem,
            [],
            0,
            threshold,
            0,0
        )
        self.ranked.sort(key=lambda x: x[2])
        """
        self.ranked = [
        (rank, *entry[1:])
        for rank, entry in enumerate(self.ranked, start=1)
        ]
        """
        return self.best, self.worst  , self.ranked
          
    def dfs_recursive(
        self, 
        weights,
        no_elem,
        clique,
        start,
        threshold,
        node_sum,
        edge_sum
        ):

        # Not enough codes left to complete this clique
        if len(clique) + (len(self.codeids) - start) < no_elem:
            return False

        no_edges = no_elem * (no_elem - 1) / 2
        #Evaluate a branch if its full 
        if len(clique) == no_elem:
            node_mean = node_sum / no_elem
            if no_edges == 0:
                edge_mean = 0 
            else:
                edge_mean = edge_sum / no_edges
            total = float(node_mean + edge_mean)

            if total > self.worst_total:
                self.worst_total = total
                self.worst = clique
                
            if total < self.best_total:
                self.best_total = total
                self.best = clique

            # Stop if threshold reached
            if threshold is not None and total < threshold:
                return True

            self.ranked.append((node_mean, edge_mean, total, clique))
            return False

        #Evaluate a partial branch, prune if above threshold

        # Use sums passed from previous recursion
        if no_edges==0:
            lower_bound = node_sum/no_elem
        else:
            lower_bound = (
                node_sum / no_elem
                + edge_sum / no_edges
            )
        # Prune this branch if it can no longer meet the threshold
        if threshold is not None and lower_bound >= threshold:
            return False
            
        #Depth-first, check one branch until the end before going to next
        for i in range(start, len(self.codeids)):
            code = int(self.codeids[i])

            # Add only the NEW node
            new_node_sum = node_sum + weights.node[code]
        
            # Add only the NEW edges
            new_edge_sum = edge_sum
    
            for other in clique:
                new_edge_sum += weights.edge[f"{other}-{code}"]

            found = self.dfs_recursive(
                weights, 
                no_elem,
                clique + [code],
                i+1,
                threshold,
                new_node_sum,
                new_edge_sum
            )
            if found: return True
        return False






#PLOTTER!

    def plot(self, output, cliquetype="best"):
        if output == "SF":
            
            for j in range(len(self.allcodes)):
                plt.rcParams['font.size'] = 10
                plt.figure()
                plt.plot(self.sf_plot[j])
                plt.title(f"c{self.codeids[j]} = {self.allcodes[self.codeids[j]]}")
                plt.xlabel("Discrete bin number")
                plt.ylabel("Normalised Power")
                plt.grid()
                plt.text(
                    0.02, 0.95,
                    f"SF={self.sfs[j]:.2f}",
                    transform=plt.gca().transAxes,
                    ha='left',
                    va='top',
                    bbox=dict(facecolor='white', edgecolor='black')
                )
                plt.savefig(f"./results/{self.codefam}/sf/sf_code{self.codeids[j]}.png")
                plt.close()
        if output == "ACF":
            for j in range(len(self.allcodes)):
                plt.rcParams['font.size'] = 10
                acf = self.acfplot[j][0]
                height = self.acfplot[j][2]
                lips = np.interp(self.acfplot[j][3], np.arange(len(acf)), self.lags) 
                rips = np.interp(self.acfplot[j][4], np.arange(len(acf)), self.lags) 

                #plot and save images 
                plt.figure()
                plt.plot(self.lags, acf)
                plt.xlabel("Lag")
                plt.ylabel("ACF")
                plt.title(f"Code Index {self.codeids[j]} = {self.allcodes[self.codeids[j]]}")
                zerolag = len(acf)//2
                plt.plot(self.lags[zerolag], acf[zerolag], "x")
                plt.hlines(height, lips, rips, color="C2")  #yvalue, xmin, xmax 

                plt.text(
                    0.02, 0.95,
                    f"ACF width={self.acfs[j]:.2f}",
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
                plt.rcParams['font.size'] = 10

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
                plt.title(f" code[{ci}] x code[{cj}] \n {self.allcodes[ci]} x {self.allcodes[cj]}")

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

            plt.rcParams['font.size'] = 15

            G = nx.Graph()
            
            G.add_nodes_from(clique)
            G.add_edges_from(itertools.combinations(clique, 2))
            pos = nx.circular_layout(G)

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
                    1+10*self.edge_weight.get(f"{i}-{j}", self.edge_weight.get(f"{j}-{i}", 1))
                    for i, j in G.edges()
                ]            )

            nx.draw_networkx_labels(G, pos, font_size=20)
            plt.title("Complete Network | WH Lc=8")
            plt.show()


if __name__ == "__main__":
    #initiate code config
    lc = 8
    ids = np.arange(1,lc)
    WHcfg = CDMcfg(
        simtime = 1, 
        lc = lc,
        codefam="WH",
        no_elem= 4,
        codeids=ids,
        SPREADING_STAGE="IF")


    WHcodes = CodeSelector(WHcfg)

    weights = WHcodes.generate_weights()
    best, worst, ranked = WHcodes.find_best_clique(weights, WHcfg.no_elem)

    
    print("best =",best)
    print("worst = ", worst)
    print("ranked=", ranked)
    WHcodes.plot("CSD")
    #WHcodes.plot("ACF")
    #WHcodes.plot("CCF")
    #WHcodes.plot("SF")
    #WHcodes.plot("network", "best")
