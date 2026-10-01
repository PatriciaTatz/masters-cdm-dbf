import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
######## NAUGHTY CODE STRUCTURE! 

def ebno_at_ber(ebno_db, ber, ber_target):
    """
    Interpolate the Eb/N0 required to achieve ber_target.
    """

    ebno_db = np.asarray(ebno_db)
    ber = np.asarray(ber)

    # check target is within simulated BER range
    if ber_target > np.max(ber) or ber_target < np.min(ber):
        return np.nan

    idx = np.argsort(ber)

    return np.interp(
        np.log10(ber_target),
        np.log10(ber[idx]),
        ebno_db[idx]
    )


def get_rank(title):
    return int(title.split("N=")[-1])


#############

folder_name = "ex_qpsk"
fileloc = Path(f"./results/WH/raw_data/thesis_data/{folder_name}")
ber_data = {}

for file_path in fileloc.glob("*.npy"):
    title = file_path.stem
    if title == "theory" or title == "ebno":
        continue
    ber_data[title] = np.load(file_path)

"""

ber_data = dict(sorted(
    ber_data.items(),
    key=lambda item: get_rank(item[0]),
    reverse=True
))

"""

ebno_db = np.load(f"./results/WH/raw_data/thesis_data/ebno.npy")
ber_theory = np.load(f"./results/WH/raw_data/thesis_data/theory.npy")

ber_target = 1e-5
# Reference Eb/No from theoretical curve
ebno_ref = ebno_at_ber(
        ebno_db,
        ber_theory,
        ber_target
    )

print(f"Theoretical Eb/No @ BER={ber_target}: {ebno_ref:.2f} dB")

plt.figure(figsize=(10,9))
plt.style.use("classic")
colors = plt.get_cmap("tab20b")(np.linspace(0, 1, len(ber_data)))
plt.rcParams['font.size'] = 10
deg = [] 
#cmap = plt.get_cmap('turbo')
#max_rank = len(ber_data)
#ADD THEORY
plt.semilogy(
    ebno_db,
    ber_theory,
    linestyle='-',
    color='black',
    marker='*',
    linewidth=1.5,
    markersize=6,
    label='Theoretical-BPSK'
)
i =0
for title, ber in ber_data.items():
    #rank = get_rank(title)
    #color = cmap((rank - 1) / (max_rank - 1))
    print(title)

    #Calculate degradation
    ebno_meas = ebno_at_ber(
        ebno_db[:len(ber)],
        ber,
        ber_target
    )
    degradation = ebno_meas - ebno_ref

    if np.isnan(degradation):
        deg.append(1e6)
    else:
        deg.append(degradation)
    print(
        f"{title}: "
        f"Eb/No={ebno_meas:.2f} dB, "
        f"Degradation={degradation:.2f} dB"
    )

    #Plot data
    #rank =int(title.split("Rank = ")[-1]) 
    #color = cmap((rank - 1) / (len(ber_data) - 1))
    plt.semilogy(
        ebno_db[:len(ber)],
        ber,
        linewidth=2,
        markersize=3,
        #color=colors[i],
        label=rf"{title}"
    )
    i+=1


# Show only 10^n labels
ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
ax.yaxis.set_minor_formatter(NullFormatter())

plt.title("Avg. BER per Path\n [IF-Spread, Bm=Bs+Δf, WH, Lc=8, QPSK]", fontsize=18)
plt.xlabel("Eb/No (dB)", fontsize=20)
plt.ylabel("Bit Error Rate", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True, which='both', linestyle='-', alpha=0.6)
plt.legend(loc='lower left', fontsize=15)

plt.show()



