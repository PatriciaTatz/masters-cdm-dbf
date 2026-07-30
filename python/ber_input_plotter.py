import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


ber_titles = [
    'Rx with MF only',
    'Rx only', 
    ]

ber_files = [
    'noCDM.npy',
    'noRx.npy'
]


fileloc = "./results/WH/raw_data/plot"
ebno_db = np.load(f"{fileloc}/ebno.npy")
ber_theory = np.load(f"{fileloc}/theory.npy")


plt.figure()


print(len(ebno_db))

for i in range(len(ber_titles)):
    ber_data = np.load(f"{fileloc}/{ber_files[i]}")

    print(ber_titles[i])
    print(ber_data)
    plt.semilogy(
        ebno_db[:len(ber_data)],
        ber_data,
        linestyle='--',
        marker='o',
        linewidth=2,
        markersize=6,
        label=ber_titles[i]
    )
plt.title('Calibration Plots')
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

plt.xlabel("Eb/No (dB)")
plt.ylabel("Bit Error Rate")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.show()
