
# Tommy Sonne Alstrøm
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class FilterBank:
    def __init__(self, numtaps, fs, numbanks, offset=0):
        # check inputs
        assert numbanks >= 2, 'Number of banks must be at least 2'
        assert fs > 0, 'Sampling frequency must be greater than 0'
        assert numtaps > 0, 'Number of taps must be greater than 0'
        assert numtaps % 2 == 1, 'Number of taps must be odd'

        self.numtaps = numtaps
        self.fs = fs
        self.numbanks = numbanks
        self.banks = []
        self.bandwidth = (fs / 2) / numbanks

        # create filterbank
        # end bands are 0 and fs/2
        # create first filter as low pass
        h = signal.firwin(numtaps, self.bandwidth, fs=fs, pass_zero='lowpass')
        self.banks.append(h)
        band_start = self.bandwidth + offset

        for _ in range(1, numbanks - 1):
            h = signal.firwin(numtaps, [band_start, band_start + self.bandwidth], fs=fs, pass_zero='bandpass')
            self.banks.append(h)
            band_start += self.bandwidth

        # create last filter as high pass
        h = signal.firwin(numtaps, band_start, fs=fs, pass_zero='highpass')
        self.banks.append(h)

    def plot_filter_bank(self):
        fig, ax = plt.subplots(2, 1, figsize=(16, 10))
        for h in self.banks:
            w, H = signal.freqz(h, 1, worN=2000)
            H = abs(H)
            ax[0].plot((self.fs * 0.5 / np.pi) * w, H)

            H_db = 20 * np.log10(H)
            ax[1].plot((self.fs * 0.5 / np.pi) * w, H_db)

        ax[0].set_title('Magnitude Response')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Gain')
        ax[1].set_ylabel('Gain (dB)')

        ax[0].grid(True)

        ax[1].grid(True)
        fig.tight_layout()
        fig.show()

    def apply_filter_bank(self, x, plot=False):
        y = np.zeros((len(x), self.numbanks))
        for inx, h in enumerate(self.banks):
            y[:, inx] = signal.lfilter(h, 1, x)

        if plot:
            # plot each individual filter output
            fig, ax = plt.subplots(self.numbanks, 1, figsize=(16, 10))
            for inx in range(self.numbanks):
                title = f'Filter {inx} - band: {inx*self.bandwidth} - {(inx+1)*self.bandwidth} Hz'
                ax[inx].plot(y[:, inx])
                ax[inx].set_title(title)
                ax[inx].grid(True)

            fig.tight_layout()
            fig.show()
        return y

    def forward(self, x):
        # apply filterbank on the signal
        y = self.apply_filter_bank(x)
        return np.sum(y, axis=1)


def main():
    # 1 sec, Fs = 8000 Hz,
    numtaps = 611
    grp_delay = (numtaps - 1) // 2  # filter is symmetric
    n = 200  # number of points
    Fs = 200


    # create filterbank and plot
    numfilters = 100
    filterbank = FilterBank(numtaps, Fs, numfilters)
    filterbank.plot_filter_bank()
    return 

    f1 = 2.5  # frequency of the first component
    f2 = 6.5  # frequency of the second component

    t = np.arange(n) / Fs  # time axis
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)  # time-domain signal

    # add f3 from t=1 to t=2
    f3 = 8.5
    x[Fs:Fs * 3] += np.sin(2 * np.pi * f3 * t[Fs:Fs * 3])

    # apply filterbank
    y = filterbank.apply_filter_bank(x, plot=True)
    mask = np.ones(numfilters)

    # example - no mask (the signal is identical to original)
    y_filter = np.sum(y * mask, axis=1)
    y_filter = y_filter[grp_delay:]
    y_filter = np.pad(y_filter, (0, grp_delay), 'constant')

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.plot(t, x, 'b')
    ax.plot(t, y_filter, 'r')
    ax.grid(True)
    fig.show()

    # # example mask - mask out the last 4 filters
    mask[-5:-1] = 0
    # # plot signal and filtered signal
    # y_filter = np.sum(y * mask, axis=1)
    # y_filter = y_filter[grp_delay:]
    # y_filter = np.pad(y_filter, (0, grp_delay), 'constant')

    # fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    # ax.plot(t, x, 'b')
    # ax.plot(t, y_filter, 'r')
    # ax.grid(True)
    # fig.show()


if __name__ == '__main__':
    main()