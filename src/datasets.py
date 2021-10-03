import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def _generate_example(random_seed, simulator, label_encoder):
    data = simulator.generate(random_seed)
    return data['chromatogram'], label_encoder.encode(data['loc'], data['area'])

def generate_dataset(n, simulator, label_encoder):
    func = partial(_generate_example, simulator=simulator, label_encoder=label_encoder)
    with Pool() as pool:
        examples = [i for i in tqdm(pool.imap(func, range(n)), total=n)]
    x = np.stack([example[0] for example in examples], axis=0)
    y = np.stack([example[1] for example in examples], axis=0)
    return x, y
