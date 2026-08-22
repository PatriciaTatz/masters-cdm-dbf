import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import matplotlib.cm as cm
"""
ber_titles = [
    'N=1',
    'N=2',
    'N=3',
    'N=4',
    'N=5',
    'N=6',
    'N=7',
    ]


ber_files = [
    'N1L8.npy',
    'N2L8.npy',
    'N3L8.npy',    
    'N4L8.npy',
    'N5L8.npy',
    'N6L8.npy',
    'N7L8.npy',
]


ber_titles = [
    'N=2, L=32 Optimised WH',
    'N=2, L=32 Optimised Gold'
]


ber_files = [
    './results/WH/raw_data/plot/N2L32.npy',
    './results/Gold/raw_data/plot/N2L32.npy'
]
"""


ber_titles = [
    'N=1',
    'N=2',
    'N=3',
    'N=4',
    'N=5',
    'N=6',
    ]

ber_files = [
    'N1L7.npy',
    'N2L7.npy',
    'N3L7.npy',    
    'N4L7.npy',
    'N5L7.npy',
    'N6L7.npy'
]



plt.rcParams['font.size'] = 20
fileloc = "./results/WH/raw_data/plot"
ebno_db = np.load(f"{fileloc}/ebno.npy")
ber_theory = np.load(f"{fileloc}/theory.npy")
colors = cm.jet(np.linspace(0, 1, len(ber_titles)))

plt.figure()
#avgloss = [] 
filedata = "./results/Gold/raw_data/plot/"
for i in range(len(ber_titles)):
    print(ber_titles[i])
    ber_data = np.load(f"{filedata + ber_files[i]}")
    target_BER = 1e-5
    """
    snr_model = np.interp(
        np.log10(target_BER),
        np.log10(ber_data[::-1]),
        ebno_db[::-1]
    )
    snr_theory = np.interp(
        np.log10(target_BER),
        np.log10(ber_theory[::-1]),
        ebno_db[::-1]
    )
    
    implementation_loss = snr_model - snr_theory

    #print(f"SNR (Theory): {snr_theory} dB")
    #print(f"SNR (Model) : {snr_model} dB")
    #print(f"Loss : {implementation_loss} dB")       
    """
    plt.semilogy(
        ebno_db[:len(ber_data)],
        ber_data,
        linestyle='--',
        marker='o',
        linewidth=2,
        markersize=6,
        color=colors[i],
        label=ber_titles[i]
    )



plt.title('Rx Signal on Element Path=1')
plt.semilogy(
    ebno_db,
    ber_theory,
    linestyle='-',
    color='black',
    marker='*',
    linewidth=2.5,
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
