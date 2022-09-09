"""Configuration."""

ROOTDIR: str = "dataset"

# Entry point of the dataset
DATASET: str = ROOTDIR + "/dataset.json"

# Minimum number of szimuth bins
NUMBER_AZIMUTH_BINS_MIN: int = 32

# Minimum number of elevation bins
NUMBER_ELEVATION_BINS_MIN: int = 64

# Minimum number of doppler bins
NUMBER_DOPPLER_BINS_MIN: int = 16

# Minimum number of range bins
NUMBER_RANGE_BINS_MIN: int = 256

# DoA estimation methods
# values: "fft", "esprit"
DOA_METHOD: str = "esprit"
