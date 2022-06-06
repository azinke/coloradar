"""Radar Digital Signal Processing.

This radar signal processing module provides the necessary tools to
process the raw IQ measurements of a MIMO radar sensor into exploitable
pointcloud and heatmap.

NOTE: Make sure that a calibration stage is applied to the raw ADC data
before further processing.
"""
from random import sample
from typing import Optional
import numpy as np
from numpy.fft import fft


# Speed of light
C: float = 299792458.0


def fft_size(size: int) -> int:
    """Computed the closest power of 2 to be use for FFT computation.

    Argument:
        size: current size of the samples.

    Return:
        Adequate window size for FFT.
    """
    return 2 ** int(np.ceil(np.log(size) / np.log(2)))



def get_range_resolution(ns: int, fs: float, fslope,
                        is_adc_filtered: bool = True) -> float:
    """Compute the range resolution of a Radar sensor.

    Arguments:
        ns: Number of ADC samples per chirp
        fs: Sampling frequency
        fslope: Chrip frequency slope
        is_adc_filtered: Boolean flag to indicate if a window function
        has been applied to the ADC data before processing. In such case
        the range resolution is affected. This parameter is set to True
        by default.

    Return:
        Range resolution in meter
    """
    rres: float = C / (ns * fslope / fs)
    if not is_adc_filtered:
        rres = rres / 2
    return rres


def get_velocity_resolution(nc: int, fstart: float, tc: float,
                            is_adc_filtered: bool = True) -> float:
    """Compute the vlocity resolution of a Radar sensor.

    Arguments:
        nc: Number of chirps per frame
        fstart: Start frequency of the chirp
        tc: Chirp time
            tc = Idle time + End time
        is_adc_filtered: Boolean flag to indicate if a window function
            has been applied to the ADC data before processing. In such case
            the velocity resolution is affected. This parameter is set to True
            by default.

    Return:
        Range velocity resolutio n in meter/s
    """
    lbd: float = C / fstart # lambda
    vres = lbd / (tc * nc)
    if not is_adc_filtered:
        vres = vres / 2
    return vres



def get_range_bins(ns: int, fs: float, fslope) -> np.array:
    """Return the range bins.

    Arguments:
        ns: Number of ADC samples per chirp
        fs: Sampling frequency
        fslope: Chrip frequency slope

    Return:
        Array of range bins
    """
    rmax: float = fs * C / (2 * fslope)
    # Resolution used for rendering
    # Note: Not the actual sensor resolution
    rres = rmax / ns
    bins = np.arange(0, rmax, rres)
    return rres * np.arange(ns)


def get_velocity_bins(ntx: int, nrx: int, nc: int, fstart: float,
        tc: float)  -> np.array:
    """Compute the velocity bins

    Arguments:
        ntx:ntx: Number of transmission antenna
        nrx: Number of reception antenna
        nc: Number of chirps per frame
        fstart: Start frequency of the chirp
        tc: Chirp time
            tc = Idle time + End time

    Return:
        Array of velocity bins
    """
    vmax: float = (C / fstart) / (4.0 * tc * ntx)
    # Resolution used for rendering
    # Not the actual radar resolution
    vres = vmax / nc

    bins = np.arange(-vmax/2, vmax/2, vres)
    return bins


def get_elevation_bins() -> np.array:
    """."""
    pass


def get_azimuth_bins() -> np.array:
    """."""
    pass


def range_fft(adc_samples: np.array, ns: int,
             coupling: Optional[np.array] = None, filter: bool = True) -> np.array:
    """Perform the Range Fast Fourier Transform of a MIMO Radar.

    Arguments:
        adc_samples: Calibrated raw ADC samples
                    The samples are expected to be a data cube of shape:
                    (ntx, nrx, nc, ns)
                        ntx: Number of transmission antenna
                        nrx: Number of reception antenna
                        nc: Number of chirps per frame
                        ns: Number of samples per chirp
        ns: Number of samples per chirp
        coupling: Optional antenna coupling correction on the Range-FFT
        filter: Boolean flag to apply Blackman window function on the data
                before FFT. It's enabled by default.

    Return:
        Range-FFT array
    """
    if filter:
        # Filter the ADC samples with Blackman window function in order to
        # reduce sidelobes
        # NOTE: This will induce a reduced range, doppler and angle resolution
        adc_samples *= np.blackman(ns).reshape(1, 1, -1)
    return fft(adc_samples, ns, -1) - coupling


def doppler_fft(rfft: np.array, ntx, nrx, nc) -> None:
    """Perform the Doppler Fast Fourier Transform of a MIMO Radar.

    Arguments:
        rfft: Range-FFT
              Result obtained from the Range-FFT of the ADC samples
        ntx: Number of transmission antenna
        nrx: Number of reception antenna
        nc: Number of chirps per frame

    Return:
        Doppler-FFT array
    """
    data = rfft.reshape(ntx * nrx, nc, -1)
    dfft = fft(data, nc, -2)
    return dfft


def angle_fft(dfft: np.array, ntx, nrx) -> np.array:
    """Perform the Angle Fast Fourier Transform of a MIMO Radar.

    Arguments:
        dfft: Doppler-FFT
            Result obtained from the Doppler-FFT stage
        ntx: Number of transmission antenna
        nrx: Number of reception antenna

    Return:
        Angle-FFT
    """
    afft = fft(dfft.reshape(ntx * nrx, -1), ntx * nrx, 0)
    return afft

def range_nci(dfft: np.array) -> np.array:
    """Range Non-Coherent integration.

    The integration is applied across all the virtual channels on the Doppler-FFT output.

    Arguments:
        dfft: Doppler-FFT ouput array of shape (ntx*nrx, nc, ns)
            ntx: Number of transmission antenna
            nrx: Number of reception antenna
            nc: Number of chirps per frame per RX antenna
            ns: Number of samples per chirp
        

    Return:
        1D Non-Coherently integrated Doppler-FFT array
        Useful for CFAR detection
    """
    return np.sum(dfft, axis=1) # 10 * np.log10(np.abs(np.sum(dfft, axis=1)))


def doppler_nci(dfft: np.array) -> np.array:
    """Doppler Non-Coherent integration.

    The integration is applied across all the virtual channels on the Doppler-FFT output.

    Arguments:
        dfft: Doppler-FFT ouput array of shape (ntx*nrx, nc, ns)
            See the meaning of ntx and nrx bellow
            nc: Number of chirps per frame per RX antenna
            ns: Number of samples per chirp
        ntx: Number of transmission antenna
        nrx: Number of reception antenna

    Return:
        1D Non-Coherently integrated Doppler-FFT array
        Useful for CFAR detection
    """
    return np.sum(dfft, axis=-1) # 10 * np.log10(np.abs(np.sum(dfft, axis=-1)))


def non_coherent_integration(dfft: np.array, ntx: int, nrx: int) -> np.array:
    """Non-Coherent integration.

    The integration is applied across all the virtual channels on the Doppler-FFT output.

    Arguments:
        dfft: Doppler-FFT ouput array of shape (ntx*nrx, nc, ns)
            See the meaning of ntx and nrx bellow
            nc: Number of chirps per frame per RX antenna
            ns: Number of samples per chirp
        ntx: Number of transmission antenna
        nrx: Number of reception antenna

    Return:
        1D Non-Coherently integrated Doppler-FFT array
        Useful for CFAR detection

    TODO: To be deleted
    """
    ncint = 10 * np.log10(np.abs(np.sum(dfft, axis=1)) + 1)
    return ncint


def os_cfar(samples: np.array, ws: int, ngc: int = 2, tos: int = 8) -> np.array:
    """Ordered Statistic Constant False Alarm Rate detector.

    Arguments:
        samples: Non-Coherently integrated samples
        ws: Window Size
        ngc: Number of guard cells
        tos: Scaling factor

    Return:
        mask
    """
    ns: int = len(samples)
    k: int = int(3.0 * ws/4.0)
    samples = np.square(np.abs(samples))

    # Add leading and trailing zeros into order to run the algorithm over
    # the entire samples
    samples = np.append(np.zeros(ws), samples)
    samples = np.append(samples, np.zeros(ws))

    mask = np.zeros(ns)
    for idx in range(ns):
        # tcells: training cells
        pre_tcells = samples[ws + idx - ngc - (ws // 2) : ws + idx - ngc]
        post_tcells = samples[ws + idx + ngc + 1: ws + idx + ngc + (ws // 2) + 1]
        tcells = np.array([])
        tcells = np.append(tcells, pre_tcells)
        tcells = np.append(tcells, post_tcells)
        tcells = np.sort(tcells)
        if samples[ws + idx] > tcells[k] * tos:
            mask[idx] = 1
    return mask


def velocity_compensation(dfft: np.array, ntx, nrx, nc) -> None:
    """Handle the compensation of velocity induiced by the MIMO antenna.

    Arguments:
        rfft: Range-FFT
              Result obtained from the Range-FFT of the ADC samples
        ntx: Number of transmission antenna
        nrx: Number of reception antenna
        nc: Number of chirps per frame

    Return:
        Corrected Doppler-FFT array
    """
    # Number of chirp per frame in MIMO configuration
    mimo_nc: int = ntx * nrx * nc
    li = np.repeat(np.arange(ntx), nrx * nc) * np.arange(-mimo_nc/2, mimo_nc/2) / (mimo_nc * ntx)
    velocity_compensation = np.exp(-2j * np.pi * li).reshape(mimo_nc, -1)
    dfft = dfft.reshape(mimo_nc, -1) * velocity_compensation
    dfft = dfft.reshape(ntx * nrx, nc, -1)
    return dfft


def angle_estimation(dfft: np.array, ntx: int, nrx: int, nc: int,
                    alayout: dict[str, np.array]) -> np.array:
    """Angle of Arrival estimation.

    Arguments:
        dfft: Doppler-FFT
        ntx: Number of transmission antenna
        nrx: Number of reception antenna
        nc: Number of chirps per TX antenna per frame
        alayout: Dictionary holding the Antenna layout
            Keys:
                "rx": Layout of the reception antenna
                "aztx": Layout of the 
    """
    pass

def process() -> None:
    """Execute a processing pipeline on raw MIMO radar measurements."""
    pass
