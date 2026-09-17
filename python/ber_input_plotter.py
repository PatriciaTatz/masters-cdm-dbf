import numpy as np 
import matplotlib.pyplot as plt 
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

#############

ber_titles = [
    'Bs=3(1+α)Rb',
    'Bs=2(1+α)Rb',
    'Bs=(1+α)Rb',
    'Bs=Rb',
    'Bs=Rb/2'
    ]

ber_files = [
    './results/WH/raw_data/N_2_Lc_8_bw_3b.npy',
    './results/WH/raw_data/N_2_Lc_8_bw_2b.npy',
    './results/WH/raw_data/N_2_Lc_8_bw_b.npy',
    './results/WH/raw_data/N_2_Lc_8_bw_fb.npy',
    './results/WH/raw_data/N_2_Lc_8_bw_fb_2.npy',
]

#lines = ['--', '-.', ':']

fileloc = "./results/WH/raw_data/plot"
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
plt.rcParams['font.size'] = 15
deg = [] 
bw = [3*45, 2*45, 45, 30, 15]
for i in range(len(ber_titles)):
    ber_data = np.load(f"{ber_files[i]}")
    print(ber_titles[i])

    ebno_meas = ebno_at_ber(
        ebno_db[:len(ber_data)],
        ber_data,
        ber_target
    )
    degradation = ebno_meas - ebno_ref

    if np.isnan(degradation):
        deg.append(1e6)
    else:
        deg.append(degradation)
    print(
        f"{ber_titles[i]}: "
        f"Eb/No={ebno_meas:.2f} dB, "
        f"Degradation={degradation:.2f} dB"
    )

    plt.semilogy(
        ebno_db[:len(ber_data)],
        ber_data,
        linewidth=2,
        markersize=3,
        label=ber_titles[i]
    )



plt.title('N=2 using WH Lc=8')

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
plt.grid(True, which='both', linestyle='-', alpha=0.6)
plt.legend()

plt.figure()
plt.plot(bw, deg, '--o')
plt.grid()
plt.xticks(
    bw,
    ['3BW', '2BW', 'BW', 'Rb', 'Rb/2']
)
plt.ylabel("Eb/No Penalty")
plt.xlabel( "Bandwidth (MHz)")
plt.ylim(0,3)
plt.show()



