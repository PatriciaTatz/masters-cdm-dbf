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
    return int(title.split("Rank = ")[-1])


#############


fileloc = Path("./results/WH/raw_data/thesis_data/low_IF_arch_badpair")
ber_data = {}

for file_path in fileloc.glob("*.npy"):
    title = file_path.stem
    if title == "theory" or title == "ebno":
        continue
    ber_data[title] = np.load(file_path)


"""
sorted_data = sorted(
    ber_data.items(),
    key=lambda item: get_rank(item[0])
)
"""

ebno_db = np.load(f"{fileloc}/ebno.npy")
ber_theory = np.load(f"{fileloc}/theory.npy")

ber_target = 1e-2
# Reference Eb/No from theoretical curve
ebno_ref = ebno_at_ber(
        ebno_db,
        ber_theory,
        ber_target
    )

print(f"Theoretical Eb/No @ BER={ber_target}: {ebno_ref:.2f} dB")

plt.figure()
plt.rcParams['font.size'] = 10
deg = [] 
#cmap = plt.get_cmap('turbo')
#max_rank = len(ber_data)

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
        linewidth=1,
        markersize=3,
        label=rf"{title}"
    )

#ADD THEORY
plt.semilogy(
    ebno_db,
    ber_theory,
    linestyle='-',
    color='black',
    marker='*',
    linewidth=1,
    markersize=6,
    label='Theoretical-BPSK'
)


# Show only 10^n labels
ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
ax.yaxis.set_minor_formatter(NullFormatter())

plt.title("Low-IF Spreading with N=2 WH Lc=8", fontsize=15)
plt.xlabel("Eb/No (dB)", fontsize=20)
plt.ylabel("Bit Error Rate", fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='-', alpha=0.6)
plt.legend(loc='lower left', fontsize=20)

plt.show()



