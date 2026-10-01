from components import CDMx 
from dataclasses import replace

from config import CDMcfg
from codeanalysis import CodeSelector
import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt 
"""
# ==============================
# PARAMETERS 
# ==============================
Rb = 30e6
Tb = 1/Rb
symbols = 1000
totaltime = symbols*Tb
lc = 7
fileloc = f"./results/Gold/raw_data/"
ccf_lc: list[float] = [] 

for n in np.arange(2,lc+1):
    config = CDMcfg(
        fb = Rb,
        lc = lc, 
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = int(n),
        codefam="Gold",
        modType="BPSK",
        codeids=np.arange(0,lc+2),
        SPREADING_STAGE = "BB"           #Location of the spreading [IF or BB]
        )
    #create signal 

    Gold =  CodeSelector(config)
    weights = Gold.generate_weights()
    best, worst, ranked =  Gold.find_best_clique(
                weights, 
                config.no_elem
                )

    #initiate and assemble receiver generate codes 
    config_best = replace(config, codeids=best)
    print(config_best)
    cdmx = CDMx(config_best)       
    codes = cdmx.generate_codes()         #spreading codes
    ccf = []

    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            cij = np.correlate(codes[i], codes[j], mode='full')
            cii =np.correlate(codes[i], codes[j], mode='full')
            print(cii[len(cij) // 2] )
            print(cij[len(cij) // 2] )
            cij = cij[len(cij) // 2] / lc
            ccf.append(cij)

    ccf_sum = np.sum(np.abs(ccf)**2)
    print(f"N={n} its ccf is ={ccf_sum}")
    
    # Convert total pairwise CCF to average per element
    ccf_avg = 2 * ccf_sum / len(codes)

    ccf_lc.append(ccf_avg)

np.save(f"{fileloc}/CCF_Lc_{lc}",ccf_lc)
"""
#PLOTTER!

folder_name = "bb_lc_n"
fileloc = Path(f"./results/Gold/raw_data/thesis_data/{folder_name}")
data = {}

for file_path in fileloc.glob("*.npy"):
    title = file_path.stem
    if title == "theory" or title == "ebno":
        continue
    data[title] = np.load(file_path)


plt.figure(figsize=(10,9))
plt.style.use("classic")
for title, values in data.items():
    n = np.arange(2,30)[:len(values)]
    plt.plot(n, values, '-o', label=f"{title}")
    plt.grid()
plt.axhline(y=0.02, color='r', linestyle='--', label='1 dB threshold')
plt.ylim(0, 0.03)
plt.xlabel("N Elements", fontsize=18)
plt.ylabel("Total Cross Correlation Coefficient", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Element isolation at Boresight", fontsize=15)

plt.legend()
plt.show()
