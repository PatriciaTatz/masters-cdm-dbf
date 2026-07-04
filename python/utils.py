"""
utilities
--- plotter 

"""
import matplotlib.pyplot as plt


class ArraySignalsPlot:

    def subplots(signal1, title1, signal2, title2):
        fig1, ax1 = plt.subplots(2, 1)

        ax1[0].plot(signal1, '-o')
        ax1[0].set_title(title1)

        ax1[1].plot(signal2, '--o')
        ax1[1].set_title(title2)

        plt.tight_layout()
        plt.show()


