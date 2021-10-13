# Deep learning models for peak detection in chromatography

## How to train the Convolutional Neural Network (CNN)

## How to perform peak detection with the CNN

## How to perform peak detection with alternative approach

```python
simulator = Simulator(
    resolution=16384,
    num_peaks_range=(1, 100),
    snr_range=(5.0, 20.0),
    amplitude_range=(25, 250),
    loc_range=(0.05, 0.95),
    scale_range=(0.001, 0.003),
    asymmetry_range=(-0.1, 0.1),
    noise_type='white',
)

# simulator.sample(indices=...) will return an iterator which yields (per iteration)
# a dictionary with the following keys: 'chromatogram', 'loc', 'scale', 'amplitude'
# and 'area'. Each index in indices will directly be used as the random seed for
# the generator method (simulator._generate(random_state=index)).
test_samples = simulator.sample(indices=range(110_000, 120_000), verbose=1)

# loop over each example in iterator (this may take a few minutes to iterate over
# as we're iterating over 10,000 examples)
outputs = []
for sample in test_samples:
  # obtain chromatogram and perform some peak detection on it
  output = peak_detection_fn(sample['chromatogram']))
  # accumulate output
  outputs.append(output)

# Do additional stuff do obtain ROC-AUC and MRE values, which can be compared with CNN
```
