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
DOA_METHOD: str = "esprit"

# Radar Digital Signal Processing Method
# values: "normal", "fesprit", "tdp"
#   - "normal" stands for chain of Range-Doppler FFT, CFAR and DOA
#   - "fesprit" also computes the Range-Doppler FFT but uses ESPRIT for
#     a precise frequency estimation for Range, Doppler and DOA. No need
#     for CFAR
#   - "tdp" stands for time domain processing
# NOTE: "fesprit" is still being tested and optimized
#
RDSP_METHOD: str = "normal"


# 2D Range-Doppler OS-CFAR Parameters used for generating
# radar pointclouds
RD_OS_CFAR_WS: int = 8         # Window size
RD_OS_CFAR_GS: int = 1         # Guard cell
RD_OS_CFAR_K: float = 0.75     # n'th quantile
RD_OS_CFAR_TOS: int = 8        # Tos factor
