import numpy as np
import pandas as pd
from tqdm import tqdm


def gaussian_peak(x, amplitude, loc, scale):
    return amplitude * np.exp((-(x - loc)**2) / (2 * scale**2))

def asymmetrical_gaussian_peak(x, amplitude, loc, scale, asymmetry):
    # assymetrical_gaussian_peak(..., assymetry=0) is the same as gaussian_peak(...)
    # add additional asymmetries by adding to the denominator inside np.exp:
    #   scale + asymmetry*(x-loc) + asymmetry*(x-loc)**2 ...,
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

def apply_baseline_drift(chromatogram, resolution):

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
        resolution=16384,
        num_peaks_range=(1, 100),
        snr_range=(5.0, 20.0),
        amplitude_range=(25, 250),
        loc_range=(0.05, 0.95),
        scale_range=(0.001, 0.003),
        asymmetry_range=(-0.1, 0.1),
        noise_type='white',
    ):
        self.resolution = resolution
        self.num_peaks_range = num_peaks_range
        self.snr_range = snr_range
        self.amplitude_range = amplitude_range
        self.loc_range = loc_range
        self.scale_range = scale_range
        self.asymmetry_range = asymmetry_range

        if noise_type == 'white':
            self.apply_baseline_noise = apply_white_noise
        elif noise_type == 'pink':
            self.apply_baseline_noise = apply_pink_noise
        else:
            raise ValueError(f"noise_type '{noise_type}' not in list of " +
                              "available noise types: ['white', 'pink']")

    def sample(self, indices, verbose=0):
        if verbose:
            indices = tqdm(indices)
        for i in indices:
            yield self._generate(i)

    def _generate(self, random_seed):

        np.random.seed(random_seed)

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
        chromatogram = self.apply_baseline_noise(chromatogram, amplitudes, snr)

        # Apply baseline drift to chromatogram
        chromatogram = apply_baseline_drift(chromatogram, self.resolution)

        return {
            'chromatogram': chromatogram,
            'loc': locs,
            'scale': scales,
            'amplitude': amplitudes,
            'area': areas,
        }
