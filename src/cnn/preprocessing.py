import numpy as np


class LabelEncoder:

    def __init__(self, num_windows):
        self.num_windows = num_windows
        self.window_size = 1 / self.num_windows
        self.window_borders = np.linspace(0, 1, self.num_windows+1)
        self.window_centers = self.window_borders[1:] - self.window_size / 2
        self.num_classes = 3

    def remove_collision(self, locs, scales, amplitudes, asymmetries):
        keep = []
        occupied = np.zeros(self.num_windows)
        for i, loc in enumerate(locs):
            dist = loc - self.window_centers
            index = np.argmin(np.abs(dist))
            if occupied[index] == 0:
                keep.append(i)
                occupied[index] = 1
        keep = np.array(keep)
        return locs[keep], scales[keep], amplitudes[keep], asymmetries[keep]

    def encode(self, locs, areas):
        labels = np.zeros((self.num_windows, self.num_classes))
        for i, (loc, area) in enumerate(zip(locs, areas)):
            # Obtain location in window (local scale)
            distance = loc - self.window_centers
            index = np.argmin(np.abs(distance))

            loc = 0.5 + distance[index] / self.window_size

            labels[index, 0] = 1.
            labels[index, 1] = loc
            labels[index, 2] = area

        return labels

    def decode(self, preds, threshold=0.5):
        # Unpack predictions: (N, num_classes) -> (N,) x num_classes
        probs, locs, areas = [
            split[:, 0] for split in np.split(preds, self.num_classes, axis=-1)
        ]
        # Where probability of peak is above threshold, obtain indices
        indices = np.where(probs > threshold)[0]
        # Transform locs and widths to global scale
        locs = locs[indices] * self.window_size + self.window_borders[indices]
        return probs[indices], locs, areas[indices]
