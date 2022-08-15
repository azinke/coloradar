"""Radar Digital Signal Processing.

This radar signal processing module provides the necessary tools to
process the raw IQ measurements of a MIMO radar sensor into exploitable
pointcloud and heatmap.

NOTE: Make sure that a calibration stage is applied to the raw ADC data
before further processing.
"""
import numpy as np


# Speed of light
C: float = 299792458.0


def steering_vector(txl: np.array, rxl: np.array,
                            az: float, el: float,) -> np.array:
    """Steering vector.

    Arguments:
        txl: TX Antenna layout
        rxl: RX Antenna layout
        az: Azimuth angle
        el: Elevation angle
    """
    # Virtual antenna array steering vector
    svect = np.zeros(len(txl) * len(rxl), dtype=np.complex128)

    # *idx: index of the antenna element
    # *az: azimuth of the antenna element
    # *el: elevation of the antenna element
    for tidx, taz, tel in txl:
        for ridx, raz, rel in rxl:
            svect[ridx + tidx * len(rxl)] = np.exp(
                1j * np.pi * (
                    (taz+raz)* np.cos(az) * np.sin(el) + (tel+rel) * np.cos(el)
            ))
    return svect


def music(signal: np.array, txl: np.array, rxl: np.array,
          az_bins: np.array, el_bins: np.array) -> np.array:
    """MUSIC Direction of Arrival estimation algorithm.

    Arguments:
        signal: Signal received by all the antenna element
                Is expected to be the combined received signal on each antenna
                element.
        txl: TX Antenna layout
        rxl: RX Antenna layout
        az_bins: Azimuth bins
        el_bins: Elevation bins
    """
    # Number of targets expected
    T: int = 1

    N = len(signal)
    signal = np.asmatrix(signal)
    # Covariance of the received signal
    R = (1.0 / N) * signal.H * signal

    eigval, eigvect = np.linalg.eig(R)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvect = eigvect[:, idx]

    V = eigvect[:, :T]
    Noise = eigvect[:, T:]

    amap = np.zeros((len(el_bins) * len(az_bins)), dtype=np.complex128)

    for eidx, el in enumerate(el_bins):
        for aidx, az in enumerate(az_bins):
            e = np.asmatrix(steering_vector(txl, rxl, az, el)).T
            amap[aidx + eidx * len(az_bins)] = 1 / np.sum(e.H * np.asmatrix(Noise))
    amap = 20 * np.log10(np.abs(amap))
    return amap


def virtual_array(adc_samples: np.array,
                  txl: list[list[int]],
                  rxl: list[list[int]]) -> np.array:
    """Generate the virtual antenna array matching the layout provided.

    Arguments:
        adc_samples: Raw ADC samples with the shape (ntx, nrx, nc, ns)
                        ntx: Number of TX antenna
                        nrx: Number of RX antenna
                        nc: Number of chirps per frame
                        ns: Number of samples per chirp
        txl: TX antenna layout array
                - Structure per row: [tx_idx, azimuth, elevation]
                - Unit: Half a wavelength
        rxl: RX antenna layout array
                - Structure: [tx_idx, azimuth, elevation]
                - Unit: Half a wavelength

    Return:
        The virtual antenna array of shape (nel, naz, nc, ns)
            nel: Number of elevation layers
            naz: Number of azimuth positions
            nc, ns: See above (description of `adc_samples`)

        See the variable `va_shape` to see how the shape is estimated
    """
    _, _, nc, ns = adc_samples.shape

    # Shape of the virtual antenna array
    va_shape: tuple = (
        # Length of the elevation axis
        # the "+1" is to count for the 0-indexing used
        np.max(txl[:, 2]) + np.max(rxl[:, 2]) + 1,

        # Length of the azimuth axis
        # the "+1" is to count for the 0-indexing used
        np.max(txl[:, 1]) + np.max(rxl[:, 1]) + 1,

        # Number of chirps per frame
        nc,

        # Number of samples per chirp
        ns,
    )

    # Virtual antenna array
    va = np.zeros(va_shape, dtype=np.complex128)

    # *idx: index of the antenna element
    # *az: azimuth of the antenna element
    # *el: elevation of the antenna element
    for tidx, taz, tel in txl:
        for ridx, raz, rel in rxl:
            # When a given azimuth and elevation position is already
            # populated, the new value is added to the previous to have
            # a strong signal feedback
            va[tel+rel, taz+raz, :, :] += adc_samples[tidx, ridx, :, :]
    return va


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
    return np.arange(0, rmax, rres)


def get_velocity_bins(ntx: int, nv: int, fstart: float, tc: float) -> np.array:
    """Compute the velocity bins

    Arguments:
        ntx:ntx: Number of transmission antenna
        nv: Number of expected velocity bins
        fstart: Start frequency of the chirp
        tc: Chirp time
            tc = Idle time + End time

    Return:
        Array of velocity bins
    """
    vmax: float = (C / fstart) / (4.0 * tc * ntx)
    # Resolution used for rendering
    # Not the actual radar resolution
    vres = vmax / nv

    bins = np.arange(-vmax/2, vmax/2, vres)
    return bins


def get_elevation_bins() -> np.array:
    """."""
    pass


def get_azimuth_bins() -> np.array:
    """."""
    pass


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


class ObjectDetected:
    """Object detected.

    Definition of object detected by applying CFAR

    NOTE: It's possible to have multiple detections on the same object
    depending on the resolution of the radar sensor
    """

    vidx: int = -1      # Velocity bin index
    ridx: int = -1      # Range bin index
    aidx: int = -1      # Azimuth bin index
    eidx: int = -1      # Elevation bin
    snr: float = 0      # Signal over Noise ratio

    def __str__(self) -> str:
        return f"Obj(SNR:{self.snr:.2f})"

    def __repr__(self) -> str:
        return self.__str__()


def nq_cfar_2d(samples, ws: int, ngc: int,
             quantile: float = 0.75, tos: int = 8) -> np.array:
    """N'th quantile statistic Constant False Alarm Rate detector.

    The principle is exactly the same as the Ordered Statistic
    Constant False Alarm Rate detector. This routine just applies
    it on a 2D signal.

    Arguments:
        samples: 2D signal to filter
        ws (int): Window size
        ngc (int): Number of guard cells
        quantile (float): Order of the quantile to compute for the noise
            power estimation
        tos (int): Scaling factor for detection an object
    """
    nx, ny = samples.shape
    mask = np.zeros((nx, ny))
    detections: list[ObjectDetected] = []

    for xidx in range(nx):
        # Before CUT (Cell Under Test) start index on the x-axis
        xbs: int = xidx - ws
        xbs = xbs if (xbs > 0) else 0

        # Before CUT (Cell Under Test) end index on the x-axis
        xbe: int = xidx - ngc
        xbe = xbe if (xbe > 0) else 0

        # After CUT (Cell Under Test) start index on the x-axis
        xas: int = xidx + ngc + 1
        # After CUT (Cell Under Test) end index on the x-axis
        xae: int =  xidx + ws + 1
        xae = xae if (xae < nx) else nx

        for yidx in range(ny):
            # Before CUT (Cell Under Test) start index on the y-axis
            ybs: int = yidx - ws
            ybs = ybs if (ybs > 0) else 0

            # Before CUT (Cell Under Test) end index on the y-axis
            ybe: int = yidx - ngc

            # After CUT (Cell Under Test) start index on the y-axis
            yas: int = yidx + ngc + 1

            # After CUT (Cell Under Test) end index on the y-axis
            yae: int =  yidx + ws + 1
            yae = yae if (yae < ny) else ny

            tcells = np.array([])
            if xbe > 0:
                tcells = samples[xbs:xbe, ybs:yae].reshape(-1)

            if xas < nx - 1:
                tcells = np.append(
                    tcells,
                    samples[xas:xae, ybs:yae].reshape(-1)
                )

            if ybe > 0:
                tcells = np.append(
                    tcells,
                    samples[xbe:xas, ybs:ybe,].reshape(-1)
                )

            if yas < nx - 1:
                tcells = np.append(
                    tcells,
                    samples[xbe:xas, yas:yae,].reshape(-1)
                )
            m = np.quantile(tcells, quantile, method="weibull")
            if samples[xidx, yidx] > (m * tos):
                mask[xidx, yidx] = 1
                obj = ObjectDetected()
                obj.vidx = xidx
                obj.ridx = yidx
                obj.snr = samples[xidx, yidx] / m
                detections.append(obj)
    return mask, detections


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
    dfft = dfft.reshape(ntx, nrx, nc, -1)
    return dfft
