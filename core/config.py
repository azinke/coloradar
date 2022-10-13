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
NUMBER_RANGE_BINS_MIN: int = 128

# DoA estimation methods
# values: "fft", "esprit"
DOA_METHOD: str = "fft"

# Radar Digital Signal Processing Method
# values: "normal", "fesprit"
#   - "normal" stands for chain of Range-Doppler FFT, CFAR and DOA
#   - "fesprit" also computes the Range-Doppler FFT but uses ESPRIT for
#     a precise frequency estimation for Range, Doppler and DOA. No need
#     for CFAR
# NOTE: "fesprit" is still being tested and optimized
RDSP_METHOD: str = "normal"
