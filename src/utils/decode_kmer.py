import numpy as np


def decode_kmer(kmer):
    """
    Transform one-hot encoded version back to labels
    """
    
    kmer_length = kmer.shape[1] // 4
    reshaped = kmer.reshape(-1, kmer_length, 4)

    indices = np.argmax(reshaped, axis=2)

    nucleotides = np.array(["A", "C", "G", "T"])
    
    decoded = nucleotides[indices]

    return np.array([''.join(nucs) for nucs in decoded])