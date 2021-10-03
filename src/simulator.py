import numpy as np


def gaussian_peak(x, amplitude, loc, scale):
    return amplitude * np.exp((-(x - loc)**2) / (2 * scale**2))

def asymmetrical_gaussian_peak(x, amplitude, loc, scale, asymmetry):
    # assymetrical_gaussian_peak(..., assymetry=0) is the same as gaussian_peak(...)
    return amplitude * np.exp(-1/2*((x - loc)/(scale + asymmetry*(x - loc)))**2)

def apply_white_noise(chromatogram, apices, signal_to_noise_ratio):
    # obtain a stddev that approximately results in noise levels close to SNR
    stddev = apices.mean() / signal_to_noise_ratio / 4
    noise = np.random.normal(0, stddev, len(chromatogram))
    return chromatogram + noise

def apply_pink_noise(chromatogram, apices, signal_to_noise_ratio, num_sources=6):
    nrows = len(chromatogram)
    ncols = num_sources
    noise = np.full((nrows, ncols), np.nan)
    noise[0, :] = np.random.random(ncols)
    noise[:, 0] = np.random.random(nrows)

    cols = np.random.geometric(0.5, nrows)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=nrows)
    noise[rows, cols] = np.random.random(nrows)

    noise = pd.DataFrame(noise)
    noise.fillna(method='ffill', axis=0, inplace=True)
    noise = noise.sum(axis=1).to_numpy()
    noise = (noise - noise.mean())

    # make noise approximately match SNR
    noise = (apices.mean() / signal_to_noise_ratio) * noise / 2
    return chromatogram + noise

def apply_basline_drift(chromatogram, resolution):

    def sigmoid(x, a, b, multiplier):
        return 1 / (1 + np.exp( - (x * a + b) )) * multiplier

    x = np.linspace(-1, 1, resolution)
    baseline_drift = np.zeros(resolution, dtype='float32')
    n = 10
    for _ in range(n):
        multiplier = np.random.uniform(-500, 500)
        a = np.random.uniform(-20, 20)
        b = np.random.uniform(-20, 20)
        baseline_drift += sigmoid(x, a, b, multiplier) / n
    return chromatogram + baseline_drift


class Simulator:

    def __init__(
        self,
        resolution,
        num_peaks_range,
        snr_range,
        amplitude_range,
        loc_range,
        scale_range,
        asymmetry_range,
        pink_noise_prob
    ):
        self.resolution = resolution
        self.num_peaks_range = num_peaks_range
        self.snr_range = snr_range
        self.amplitude_range = amplitude_range
        self.loc_range = loc_range
        self.scale_range = scale_range
        self.asymmetry_range = asymmetry_range
        self.pink_noise_prob = pink_noise_prob

    def generate(self, random_state):

        np.random.seed(random_state)

        # Randomly obtain parameters of the peaks
        num_peaks = np.random.randint(*self.num_peaks_range)
        amplitudes = np.random.uniform(*self.amplitude_range, size=(num_peaks,))
        locs = np.random.uniform(*self.loc_range, size=(num_peaks,))
        scales = np.random.uniform(*self.scale_range, size=(num_peaks,))
        snr = np.random.uniform(*self.snr_range)
        asymmetries = np.random.uniform(*self.asymmetry_range, size=(num_peaks,))
        areas = np.zeros([0], dtype='float32')

        # Generate (noise-free) chromatogram
        x = np.linspace(0, 1, self.resolution)
        chromatogram = np.zeros_like(x)
        for ampl, loc, scale, assym in zip(amplitudes, locs, scales, asymmetries):
            peak = asymmetrical_gaussian_peak(x, ampl, loc, scale, assym)
            areas = np.concatenate([areas, [np.trapz(peak, dx=x[1]-x[0])]])
            chromatogram += peak

        # Apply pink noise or white noise to chromatogram
        if self.pink_noise_prob > np.random.random():
            chromatogram = apply_pink_noise(chromatogram, amplitudes, snr)
        else:
            chromatogram = apply_white_noise(chromatogram, amplitudes, snr)

        # Apply baseline drift to chromatogram
        chromatogram = apply_basline_drift(chromatogram, self.resolution)
        chromatogram =  chromatogram[:, None] # add dimension for Conv1D

        return {
            'chromatogram': chromatogram,
            'loc': locs,
            'scale': scales,
            'amplitude': amplitudes,
            'area': areas,
        }
