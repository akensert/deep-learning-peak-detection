import numpy as np
import pandas as pd
import scipy.interpolate
from tqdm import tqdm


def gaussian_peak(x, amplitude, loc, scale):
    return amplitude * np.exp((-(x - loc)**2) / (2 * scale**2))

def asymmetrical_gaussian_peak(x, amplitude, loc, s0, s1, s2=0.0, eps=1e-7):
    # assymetrical_gaussian_peak(..., assymetry=0) is the same as gaussian_peak(...)
    # add additional asymmetries by adding to the denominator inside np.exp:
    #   s0 + s1*(x-loc) + s2*(x-loc)**2 ..., sN*(x-loc)**N

    peak = np.exp(-1/2*((x - loc)/(eps + s0 + s1*(x - loc) + s2*(x - loc)**2))**2)
    peak *= amplitude

    if s1 > 0:
        out = loc - 4 * s0
        idx = np.argmin(np.abs(x - out))
        peak[:idx] = 0.0
    else:
        out = loc + 4 * s0
        idx = np.argmin(np.abs(x - out))
        peak[idx:] = 0.0

    return peak

def apply_white_noise(chromatogram, apices, signal_to_noise_ratio):
    # obtain a stddev that approximately results in noise levels close to SNR
    stddev = apices.mean() / signal_to_noise_ratio / 4
    noise = np.random.normal(0, stddev, len(chromatogram))
    return chromatogram + noise

def apply_pink_noise(chromatogram, apices, signal_to_noise_ratio, num_sources=8):
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

def apply_baseline_drift(chromatogram, resolution, multiplier_range):

    def sigmoid(x, a, b, multiplier):
        return 1 / (1 + np.exp( - (x * a + b) )) * multiplier

    x = np.linspace(-1, 1, resolution)
    baseline_drift = np.zeros(resolution, dtype='float32')
    n = 10
    for _ in range(n):
        multiplier = np.random.uniform(*multiplier_range)
        a = np.random.uniform(-20, 20)
        b = np.random.uniform(-20, 20)
        baseline_drift += sigmoid(x, a, b, multiplier) / n
    return chromatogram + baseline_drift

def random_truncated_normal(loc, scale, minimum, maximum, size):
    values = np.zeros(size)
    logical_test = values.copy().astype(bool)
    while not np.all(logical_test):
        logical_test = np.logical_and((minimum < values), (values < maximum))
        values = np.where(logical_test, values, scale * np.random.randn(size) + loc)
    return values

def random_logarithmic_uniform(minimum, maximum, size=None):
    return 10 ** np.random.uniform(np.log10(minimum), np.log10(maximum), size=size)

def random_logarithmic_randint(minimum, maximum, size=None):
    return np.round(random_logarithmic_uniform(minimum, maximum, size)).astype(int)


class Simulator:

    def __init__(
        self,
        remove_collision_fn=None,
        resolution=8192,
        num_peaks_range=(10, 100),
        snr_range=(3.0, 300.0),
        amplitude_range=(10, 100),
        loc_range=(0.05, 0.95),
        scale_range=(0.001, 0.002),
        asymmetry_range=(0.00, 0.30),
        baseline_drift_magnitude=(-5, 5), # Not used
        white_noise_prob=1.0,
    ):
        self.remove_collision_fn = remove_collision_fn
        self.resolution = resolution
        self.num_peaks_range = num_peaks_range
        self.snr_range = snr_range
        self.amplitude_range = amplitude_range
        self.loc_range = loc_range
        self.scale_range = scale_range
        self.asymmetry_range = asymmetry_range
        self.baseline_drift_magnitude = baseline_drift_magnitude
        self.white_noise_prob = white_noise_prob

    def sample_batch(self, indices, verbose=0):
        if verbose:
            indices = tqdm(indices)
        for i in indices:
            yield self._generate_example(i)

    def _get_random_scales(self, locs, n_peaks):
        scale_loc = np.random.uniform(*self.scale_range)
        scales = random_truncated_normal(
            scale_loc,                      # mean
            scale_loc/10,                   # std
            scale_loc - scale_loc/10 * 3,   # min (mean - 3std)
            scale_loc + scale_loc/10 * 3,   # max (mean + 3std)
            n_peaks                         # size
        )
        a = np.random.uniform(1, 3)
        b = 0.0 # np.random.uniform(0, 4)
        scales *= 1 + a*locs +  b*locs**2
        return scales, a, b

    def _get_random_amplitudes(self, locs, a, b, n_peaks):
        amplitudes = np.random.uniform(*self.amplitude_range, size=n_peaks)
        amplitudes /= (1 + a*locs + b*locs)
        return amplitudes

    def _get_random_locs(self, n_peaks):
        locs = np.random.uniform(*self.loc_range, size=n_peaks)
        return np.sort(locs)

    def _get_random_asymmetries(self, locs, n_peaks):
        asymmetry_loc = np.random.uniform(*self.asymmetry_range)
        s1 = random_truncated_normal(
            asymmetry_loc,                          # mean
            asymmetry_loc/10,                       # std
            asymmetry_loc - asymmetry_loc/10 * 3,   # min (mean - 3std)
            asymmetry_loc + asymmetry_loc/10 * 3,   # max (mean + 3std)
            n_peaks                                 # size
        )
        s1 *= (1 - locs)
        s2 = -np.random.uniform(np.abs(s1/2), np.abs(s1*2))
        return s1, s2

    def _generate_example(self, random_seed):

        np.random.seed(random_seed)

        n_peaks = random_logarithmic_randint(*self.num_peaks_range)

        # 1. Generate (noise-free) chromatogram
        # a. Obtain peak parameters
        locs = self._get_random_locs(n_peaks)
        scales, a, b = self._get_random_scales(locs, n_peaks)
        ampls = self._get_random_amplitudes(locs, a, b, n_peaks)
        s1s, s2s = self._get_random_asymmetries(locs, n_peaks)

        # b. Remove collisions:
        #    If two peaks fall inside the same window, remove one of the peaks
        #    Otherwise we will train the model to predict one peak (for the
        #    given heavily-overlapping peak) even though there are two peaks
        if self.remove_collision_fn is not None:
            locs, scales, s1s, s2s, ampls = self.remove_collision_fn(
                locs, scales, s1s, s2s, ampls
            )

        # 3. Generate (noise-free) chromatogram from peaks (and obtain areas)
        x = np.linspace(0, 1, self.resolution)
        areas = np.zeros([0], dtype='float32')
        chromatogram = np.zeros_like(x)
        for i, (ampl, loc, scale, s1, s2) in enumerate(zip(ampls, locs, scales, s1s, s2s)):
            peak = asymmetrical_gaussian_peak(x, ampl, loc, scale, s1, s2)
            areas = np.concatenate([areas, [np.trapz(peak, dx=x[1]-x[0])]])
            chromatogram += peak

        # 2. Apply noise to (noise-free) chromatogram
        # a. Obtain signal-to-noise ratio (levels of noise)
        snr = random_logarithmic_uniform(*self.snr_range)
        # b. Apply white noise or pink noise to chromatogram
        if np.random.random() <= self.white_noise_prob:
            chromatogram = apply_white_noise(chromatogram, ampls, snr)
        else:
            chromatogram = apply_pink_noise(chromatogram, ampls, snr)

        # distort baseline a bit
        chromatogram += np.random.randn() * 2

        # c. Apply baseline drift to chromatogram
        #chromatogram = apply_baseline_drift(
        #   chromatogram, self.resolution, self.baseline_drift_magnitude)

        return {
            'chromatogram': chromatogram,
            'loc': locs,
            'scale': scales,
            'amplitude': ampls,
            's1': s1s,
            's2': s2s,
            'area': areas,
        }
