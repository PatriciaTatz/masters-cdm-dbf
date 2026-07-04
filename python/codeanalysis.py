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
        sfs =  self.sf()
        acfs = self.acf()
        ccs = self.ccf()

        self.node_weight = {
            codeid: acf / max(sf, 1e-12)
            for codeid, sf, acf in zip(self.codeids, sfs, acfs)
        }

        for i, j, ccf_width, ccf_peak, csd_height in ccs:
            self.edge_weight[f"{i}-{j}"] += (
                1/3*csd_height +
                1/3*abs(ccf_peak) +
                1/3*ccf_width
            )

        return CodeWeights(self.node_weight, self.edge_weight)

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
        Values near:
            1 -> Has longer decay at zero lag 
            0 -> Minimal decay at zero lag
        Returns:
            acf_score : ndarray
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
        """
        Returns
        -------
        metrics : list of tuples
            (i, j, ccf_width, ccf_peak, csd_height)
        """

        self.ccf_data = []
        self.csd_data = []

        metrics = []
        zerolag = self.Lc // 2

        for i in range(1, len(self.allcodes)):
            for j in range(i + 1, len(self.allcodes)):

                # Periodic normalized CCF
                ccf = np.real(
                    np.fft.ifft(
                        np.fft.fft(self.allcodes[i]) *
                        np.conj(np.fft.fft(self.allcodes[j]))
                    )
                )
                ccf = np.fft.fftshift(ccf) / self.Lc

                widths, heights, left_ips, right_ips = peak_widths(
                    ccf,
                    [zerolag],
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

                # CSD
                fbins, csdx = csd(
                    self.allcodes[i],
                    self.allcodes[j],
                    nperseg=self.Lc,
                    return_onesided=False
                )

                self.csd_data.append(
                    (
                        self.codeids[i],
                        self.codeids[j],
                        fbins,
                        csdx,
                    )
                )

                if not isclose(fbins[1], 1 / self.Lc):
                    raise ValueError(
                        f"Expected first frequency bin = {1/self.Lc}, got {fbins[1]}"
                    )

                metrics.append(
                    (
                        self.codeids[i],
                        self.codeids[j],
                        float(widths[0]),      # CCF width
                        float(ccf[zerolag]),   # zero-lag CCF peak
                        float(np.abs(csdx[1])) # CSD amplitude at 1/Lc
                    )
                )

        width_max = max(m[2] for m in metrics)
        peak_max  = max(abs(m[3]) for m in metrics)
        csd_max   = max(m[4] for m in metrics)

        metrics_norm = [
            (
                i,
                j,
                width / width_max if width_max else 0.0,
                abs(peak) / peak_max if peak_max else 0.0,
                csd / csd_max if csd_max else 0.0,
            )
            for i, j, width, peak, csd in metrics
        ]
        return metrics_norm
        
    def find_best_clique(self, weights, no_elem):
        self.best_nodes = None
        self.worst_nodes = None 
        best_weight = float('inf')
        worst_weight = float('-inf')

        for clique in itertools.combinations(self.codeids, no_elem):
            node_sum = sum(weights.node[int(i)] for i in clique)
            edge_sum =  sum(weights.edge[f"{i}-{j}"] for i, j in itertools.combinations(clique, 2))

            total = node_sum + edge_sum
            if total < best_weight:
                best_weight = total
                self.best_nodes = [int(x) for x in clique]
            if total > worst_weight:
                worst_weight = total
                self.worst_nodes = [int(x) for x in clique]

        return self.best_nodes, self.worst_nodes
          

    def plot(self, output, cliquetype="best"):
        if output == "ACF":
            for j in range(len(self.allcodes)):
                lags = np.arange(-self.Lc//2, self.Lc//2)
                acf = self.acfplot[j][0]
                height = self.acfplot[j][2]
                lips = np.interp(self.acfplot[j][3], np.arange(self.Lc), lags) 
                rips = np.interp(self.acfplot[j][4], np.arange(self.Lc), lags) 

                #plot and save images 
                plt.figure()
                plt.plot(lags, acf)
                plt.xlabel("Lag")
                plt.ylabel("ACF")
                plt.grid()
                plt.title(f"ACF: code[{self.codeids[j]}]")

                plt.plot(lags[self.Lc//2], acf[self.Lc//2], "x")
                plt.hlines(height, lips, rips, color="C2")  #yvalue, xmin, xmax 
                plt.grid()
                plt.savefig(f"./results/{self.codefam}/acf/acf_code{j}.png")
                plt.close()
                
        elif output == "CCF":
            lags = np.arange(-self.Lc // 2, self.Lc // 2)
            for ci, cj, ccf, width, height, left_ip, right_ip in self.ccf_data:
                left_lag = np.interp(
                    left_ip,
                    np.arange(self.Lc),
                    lags
                )

                right_lag = np.interp(
                    right_ip,
                    np.arange(self.Lc),
                    lags
                )

                plt.figure()
                plt.plot(lags, ccf)
                # Zero-lag peak
                plt.plot(
                    lags[self.Lc // 2],
                    ccf[self.Lc // 2],
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

                plt.grid(True)
                plt.title(f"CCF: code[{ci}] x code[{cj}]")

                plt.savefig(
                    f"./results/{self.codefam}/ccf/ccf_codes{ci}x{cj}.png"
                )

                plt.close()

        elif output == "CSD":
            for ci, cj, fbins, csd in self.csd_data:
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

                plt.savefig(
                    f"./results/{self.codefam}/csd/csd_codes{ci}x{cj}.png"
                )
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
                node_size=[200 + v/50  for v in values],
                node_color=values,
                cmap=plt.cm.coolwarm,
            )

            nx.draw_networkx_edges(
                G,
                pos,
                width=[
                    1 + 5*self.edge_weight.get(f"{i}-{j}", self.edge_weight.get(f"{j}-{i}", 1))
                    for i, j in G.edges()
                ]
            )

            nx.draw_networkx_labels(G, pos)
            plt.show()


if __name__ == "__main__":
    #initiate code config
    WHcfg = CDMcfg(
        simtime = 1, 
        lc = 16,
        codefam="WH",
        no_elem= 5)


    WHcodes = CodeSelector(WHcfg)
    weights = WHcodes.generate_weights()
    best, worst = WHcodes.find_best_clique(weights, WHcfg.no_elem)
    print("best =",best)
    print("worst = ", worst)
    #WHcodes.plot("CCF")
    WHcodes.plot("network", "best")
