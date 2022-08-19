"""Radar.

SCRadar: Single Chip Radar Sensor
CCRadar: Cascade Chip Radar Sensor
"""
from optparse import Option
from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt

from core.calibration import Calibration, SCRadarCalibration
from core.utils.common import error, info
from core.utils import radardsp as rdsp
from .lidar import Lidar

from core.config import NUMBER_RANGE_BINS_MIN
from core.config import NUMBER_DOPPLER_BINS_MIN
from core.config import NUMBER_AZIMUTH_BINS_MIN
from core.config import NUMBER_ELEVATION_BINS_MIN


class SCRadar(Lidar):
    """Radar.

    Attrinutes:
        NUMBER_RECORDING_ATTRIBUTES: Number of 32-bit integer packed
            to form a single measurement recording
        CFAR_WS: Constant False Alarm Rate Window Size
        CFAR_GC: Constant False Alarm Rate Guard Cell
    """

    # The recorded attributes are:
    # x, y, z, I (Intensity of the reflections), Vr (Radial velocity)
    NUMBER_RECORDING_ATTRIBUTES: int = 5

    # CFAR Window size
    CFAR_WS: int = 16
    # CFAR guard cell
    CFAR_GC: int = 8

    AZIMUTH_FOV: float = np.deg2rad(60)
    ELEVATION_FOV: float = np.deg2rad(15)

    def __init__(self, config: dict[str, str],
                 calib: Calibration, index: int) -> None:
        """Init.

        Arguments:
            config (dict): Paths to access the dataset
            calib (Calibration): Calibration object (See calibration.py)
            index (int): Index of the lidar record to load
        """
        self.sensor: str = self.__class__.__name__.lower()
        self.calibration: SCRadarCalibration = getattr(calib, self.sensor)
        self.config = config
        self.index = index

        if self.sensor == "scradar":
            # Read pointcloud
            self.cld = self._load(
                index, "pointcloud", np.float32,
                (-1, self.NUMBER_RECORDING_ATTRIBUTES)
            )

        # Read heatmap
        self.heatmap = self._load(
            index, "heatmap", np.float32,
            (
                self.calibration.heatmap.num_elevation_bins,
                self.calibration.heatmap.num_azimuth_bins,
                self.calibration.heatmap.num_range_bins,
                2 # Number of value per bin (intensity and location)
            )
        )

        # read raw ADC measurements
        self.raw = None
        raw = self._load(
            index, "raw", np.int16,
            (
                self.calibration.waveform.num_tx,
                self.calibration.waveform.num_rx,
                self.calibration.waveform.num_chirps_per_frame,
                self.calibration.waveform.num_adc_samples_per_chirp,
                2 # I and Q signal measurements
            )
        )
        if raw is not None:
            # s = I + jQ
            I = np.float16(raw[:, :, :, :, 0])
            Q = np.float16(raw[:, :, :, :, 1])
            self.raw = I + 1j * Q

    def _load(self, index: int, ftype: str, dtype: np.dtype,
              shape: tuple[int, ...]) -> Optional[np.array]:
        """Load data.

        Arguments:
            index (int): Index of the datafile to load
            ftype (str): File type to load. It could be one of the following
                   values: ('pointcloud', 'heatmap', 'raw')
            dtype (np.dtype): Data type in the file
            shape (tuple): The expected shape of the ouput array. The data
                   will be reshaped based on this parameter

        Return: Data loaded. None if an error occured during the loading
                process.
        """
        filename = self._filename(
            self.config["paths"][self.sensor][ftype]["filename_prefix"],
            index,
            "bin"
        )
        filepath = os.path.join(
            self.config["paths"]["rootdir"],
            self.config["paths"][self.sensor][ftype]["data"],
            filename
        )
        try:
            data = np.fromfile(filepath, dtype)
            data = np.reshape(data, shape)
        except FileNotFoundError:
            error(f"File '{filepath}' not found.")
            data = None
        return data

    def showHeatmap(self, threshold: float = 0.15, no_sidelobe: bool = False,
                          render: bool = True) -> None:
        """Render heatmap.

        Argument:
            threshold: Value used to filter the pointcloud
            no_sidelobe: Flag used to skip the close range data from rendering
                         in order to avoid side lobes
            render: Flag triggering the rendering of the heatmap when 'true'.
        """
        if self.heatmap is None:
            info("No heatmap available!")
            return None
        ax = plt.axes(projection="3d")
        ax.set_title("Heatmap")
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Range")
        ax.set_zlabel("Elevation")
        pcl = self._heatmap_to_pointcloud(threshold, no_sidelobe)
        plot = ax.scatter(
            pcl[:, 0],
            pcl[:, 1],
            pcl[:, 2],
            c=pcl[:, 3],
            cmap=plt.cm.get_cmap(),
        )
        colorbar = plt.colorbar(plot)
        colorbar.set_label("Reflection Intensity")
        if render:
            plt.show()

    def showHeatmapBirdEyeView(self, threshold: float) -> None:
        """Render the bird eye view of the heatmp pointcloud.

        Argument:
            threshold (float): Threshold to filter the pointcloud

        TODO: To be fixed. Not working
        """
        pointcloud = self._heatmap_to_pointcloud(threshold)
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        _range_bin_width: float = self.calibration.heatmap.range_bin_width
        _num_r_num: int = self.calibration.heatmap.num_range_bins
        max_range = np.ceil(_range_bin_width * _num_r_num)

        resolution: float = _range_bin_width/10
        srange: tuple[float, float] = (-max_range, max_range)
        frange: tuple[float, float] = (0, max_range)

        ximg = (-y / resolution).astype(np.int32)
        yimg = (-x / resolution).astype(np.int32)

        ximg -= int(np.floor(srange[0] / resolution))
        yimg += int(np.floor(frange[1] / resolution))

        # Prepare the three channels of the bird eye view image
        pixels = np.zeros((len(z), 3), dtype=np.uint8)

        # Encode distance
        norm = np.sqrt(x **2 + y **2 + z**2)
        pixels[:, 0] = (255.0 / (1.0 + np.exp(-norm))).astype(np.uint8)

        # Encode radial velosity of the radar
        pixels[:, 1] = (
            255.0 / (1.0 + np.exp(-np.abs(pointcloud[:, 4])))
        ).astype(np.uint8)

        # Encode reflection intensity information
        pixels[:, 2] = (255.0 * pointcloud[:, 3]).astype(np.uint8)

        # Create the frame for the bird eye view
        # Note: the "+1" to estimate the width and height of the image is
        # to count for the (0, 0) position in the center of the pointcloud
        img_width: int = 1 + int((srange[1] - srange[0])/resolution)
        img_height: int = 1 + int((frange[1] - frange[0])/resolution)
        bev_img = np.zeros([img_height, img_width, 3], dtype=np.uint8)

        # Set the height information in the created image frame
        bev_img[yimg, ximg] = pixels

        plt.imshow(bev_img)
        plt.title(f"Heatmap BEV - ~{resolution:.4} m/pixel")
        plt.show()

    def _polar_to_cartesian(self, r_idx: int,
            az_idx: int, el_idx: int) -> np.array:
        """Convert polar coordinate to catesian coordinate.

        Example:
            self._polar_to_cartesian(range_idx - 1, az_idx - 1, el_idx - 1)
        """
        _range_bin_width: float = self.calibration.heatmap.range_bin_width
        _el_bin: float = self.calibration.heatmap.elevation_bins[el_idx]
        _az_bin: float = self.calibration.heatmap.azimuth_bins[az_idx]

        point = np.zeros(3)
        point[0] = r_idx * _range_bin_width * np.cos(_el_bin) * np.cos(_az_bin)
        point[1] = r_idx * _range_bin_width * np.cos(_el_bin) * np.sin(_az_bin)
        point[2] = r_idx * _range_bin_width * np.sin(_el_bin)
        return point

    def _to_cartesian(self, hmap: np.array) -> np.array:
        """Convert polar coordinate heatmap to catesian coordinate.

        Argument:
            hmap: The heatmap of shape (-1, 5)
                  Structure (columns):
                    [0]: Azimuth
                    [1]: Range
                    [2]: Elevation
                    [3]: Velocity
                    [4]: Intensity of reflection in dB

                @see: showHeatmapFromRaw

        Example:
            self._to_cartesian(hmap)
        """
        pcld = np.zeros(hmap.shape)
        pcld[:, 0] = hmap[:, 1] * np.cos(hmap[:, 2]) * np.cos(hmap[:, 0])
        pcld[:, 1] = hmap[:, 1] * np.cos(hmap[:, 2]) * np.sin(hmap[:, 0])
        pcld[:, 2] = hmap[:, 1] * np.sin(hmap[:, 2])
        pcld[:, 3:] = hmap[:, 3:]
        return pcld

    def _heatmap_to_pointcloud(self, threshold: float = 0.15,
                                     no_sidelobe: bool = False) -> np.array:
        """Compute pointcloud from heatmap.

        The recordings of the heatmap are stored in the polar form.
        This function prepares a pointcloud in the cartesian coordinate
        system based the heatmap

        Argument:
            threshold (float): Threshold to filter the pointcloud
            no_sidelobe: Flag used to skip the close range data from rendering
                         in order to avoid side lobes

        Return
            pcl (np.array): the heatmap point locations
        """
        _num_el_bin: int = self.calibration.heatmap.num_elevation_bins
        _num_az_bin: int = self.calibration.heatmap.num_azimuth_bins
        _num_r_num: int = self.calibration.heatmap.num_range_bins

        _range_bin_width: float = self.calibration.heatmap.range_bin_width

        # transform range-azimuth-elevation heatmap to pointcloud
        pcl = np.zeros((_num_el_bin, _num_az_bin, _num_r_num, 5))

        # Range offset to count for the side lobes of the radar self.sensor
        roffset: int = 5

        if not no_sidelobe:
            roffset = 0

        for range_idx in range(roffset, _num_r_num - 1):
            for az_idx in range(_num_az_bin - 1):
                for el_idx in range(_num_el_bin - 1):
                    _el_bin: float = self.calibration.heatmap.elevation_bins[
                        el_idx
                    ]
                    _az_bin: float = self.calibration.heatmap.azimuth_bins[
                        az_idx
                    ]
                    pcl[el_idx + 1, az_idx + 1, range_idx + 1, :3] = np.array([
                        _az_bin,                        # Azimuth
                        (range_idx) * _range_bin_width,   # Range
                        _el_bin,                        # Elevation
                    ])

        pcl = pcl.reshape(-1,5)
        pcl[:,3:] = self.heatmap.reshape(-1, 2)
        # Normalise the radar reflection intensity
        pcl[:, 3] -= np.min(pcl[:, 3])
        pcl[:, 3] /= np.max(pcl[:, 3])
        if threshold:
            pcl = pcl[pcl[:, 3] > threshold]
            # Re-Normalise the radar reflection intensity after filtering
            pcl[:, 3] -= np.min(pcl[:, 3])
            pcl[:, 3] /= np.max(pcl[:, 3])
        return pcl

    def _music(self) -> None:
        """Apply MUSIC algorithm for DoA estimation."""
        # Calibrate raw data
        adc_samples = self.raw

        # ADC sampling frequency
        fs: float = self.calibration.waveform.adc_sample_frequency

        # Frequency slope
        fslope: float = self.calibration.waveform.frequency_slope

        # Start frequency
        fstart: float = self.calibration.waveform.start_frequency

        # Ramp end time
        te: float = self.calibration.waveform.ramp_end_time

        # Chirp time
        tc: float = self.calibration.waveform.idle_time + te

        Na = 64
        Ne = 64

        if self.sensor != "scradar":
            adc_samples *= self.calibration.get_frequency_calibration()
            adc_samples *= self.calibration.get_phase_calibration()

        ntx, nrx, nc, ns = adc_samples.shape

        rfft = np.fft.fft(adc_samples, ns, -1) - self.calibration.get_coupling_calibration()
        dfft = np.fft.fft(rfft, nc, -2)
        dfft = np.fft.fftshift(dfft, -2)
        dfft = dfft.reshape(ntx * nrx, nc, ns)

        # signal = np.sum(dfft, (1, 2))
        # print("signal shape: ", signal.shape)

        vbins = rdsp.get_velocity_bins(ntx, nc, fstart, tc)
        rbins = rdsp.get_range_bins(ns, fs, fslope)

        # Azimuth bins
        ares = np.pi / Na
        abins = np.arange(-np.pi/2, np.pi/2, ares)
        # Elevation
        eres = np.pi / Ne
        ebins = np.arange(-np.pi/2, np.pi/2, eres)

        spectrum = np.zeros((ns, Ne, Na, 1), dtype=np.complex128)

        signal = np.sum(dfft, (1, 2))

        spectrum = rdsp.music(
            signal, self.calibration.antenna.txl, self.calibration.antenna.rxl, abins, ebins
        )
        '''
        hmap = np.zeros((Na * Ne, 3))

        for eidx in range(Ne):
            for aidx in range(Na):
                hmap_idx: int = aidx + Na * eidx
                hmap[hmap_idx] = np.array([
                    abins[aidx],
                    ebins[eidx],
                    spectrum[hmap_idx],
                ])
        '''

        # ax = plt.axes(projection="3d")
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_title("Test MUSIC")
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation")
        ax.set_zlabel("Gain")
        el, az = np.meshgrid(ebins, abins)
        '''
        map = ax.scatter(
            hmap[:, 0],
            hmap[:, 1],
            hmap[:, 2],
            c=hmap[:, 2],
            cmap=plt.cm.get_cmap()
        )
        '''
        surf = ax.plot_surface(
            el, az, spectrum.reshape(Ne, Na),
            cmap="coolwarm",
            rstride=1,
            cstride=1,
            alpha=None,
            # linewidth=0,
            # antialiased=False
        )
        plt.colorbar(surf, shrink=0.5, aspect=1)
        plt.show()

    def _calibrate(self) -> np.array:
        """Handle the calibration of raw ADC samples.

        Return:
            Calibrated data based on the radar sensor's type

        NOTE: Only the casacde chip radar sensor has the frequency and phase
        calibration
        """
        adc_samples = self.raw

        # Remove DC bias
        adc_samples -= np.mean(adc_samples)

        if self.sensor != "scradar":
            adc_samples *= self.calibration.get_frequency_calibration()
            adc_samples *= self.calibration.get_phase_calibration()
        return adc_samples

    def _get_fft_size(self, ne: Optional[int], na: Optional[int],
             nc: Optional[int], ns: Optional[int]) -> tuple[int, int, int, int]:
        """Get optimal FFT size.

        Arguments:
            ne: Size of the elevation axis of the data cube
            na: Size of the azimuth axis of the data cube
            nc: Number of chirp loops
            ns: Number of samples per chirp

        Return:
            Tuple of the optimal size of each parameter provided in argument
            in the exact same order.
        """
        # Estimated size of the elevation and azimuth
        if ne is not None:
            ne = rdsp.fft_size(ne)
            ne = (
                ne if ne > NUMBER_ELEVATION_BINS_MIN else NUMBER_ELEVATION_BINS_MIN
            )

        if na is not None:
            na = rdsp.fft_size(na)
            na = na if na > NUMBER_AZIMUTH_BINS_MIN else NUMBER_AZIMUTH_BINS_MIN

        if nc is not None:
            # Size of doppler FFT
            nc = rdsp.fft_size(nc)
            nc = nc if nc > NUMBER_DOPPLER_BINS_MIN else NUMBER_DOPPLER_BINS_MIN
        if ns is not None:
            # Size of range FFT
            ns = rdsp.fft_size(ns)
            ns = ns if ns > NUMBER_RANGE_BINS_MIN else NUMBER_RANGE_BINS_MIN

        return ne, na, nc, ns

    def _get_bins(self, ns: Optional[int], nc: Optional[int], na: Optional[int],
            ne: Optional[int]) ->tuple[np.array, np.array, np.array, np.array]:
        """Return the range, velocity, azimuth and elevation bins.

        Arguments:
            ne: Elevation FFT size
            na: Azimuth FFT size
            nc: Doppler FFT size
            ns: Range FFT size

        Return:
            range bins
            velocity bins
            azimuth bins
            elevation bins

        NOTE: The bins are returned in the order listed above
        """
        # Number of TX antenna
        ntx: int = self.calibration.antenna.num_tx

        # ADC sampling frequency
        fs: float = self.calibration.waveform.adc_sample_frequency

        # Frequency slope
        fslope: float = self.calibration.waveform.frequency_slope

        # Start frequency
        fstart: float = self.calibration.waveform.start_frequency

        # Ramp end time
        te: float = self.calibration.waveform.ramp_end_time

        # Chirp time
        tc: float = self.calibration.waveform.idle_time + te

        rbins = np.array([])        # Range bins
        vbins = np.array([])        # Doppler bins
        abins = np.array([])        # Azimuth bins
        ebins = np.array([])        # Elevation bins

        if ns:
            rbins = rdsp.get_range_bins(ns, fs, fslope)

        if nc:
            # Velocity bins
            vbins = rdsp.get_velocity_bins(ntx, nc, fstart, tc)

        if na:
            # Azimuth bins
            ares = 2 * self.AZIMUTH_FOV / na
            # Estimate azimuth angles and flip the azimuth axis
            abins = -1 * np.arcsin(
                np.arange(-self.AZIMUTH_FOV, self.AZIMUTH_FOV, ares) / (
                    2 * np.pi * self.calibration.d
                )
            )

        if ne:
            # Elevation
            eres = 2 * self.ELEVATION_FOV / ne
            # Estimate elevation angles and flip the elevation axis
            ebins = -1 * np.arcsin(
                np.arange(-self.ELEVATION_FOV, self.ELEVATION_FOV, eres) / (
                    2 * np.pi * self.calibration.d
                )
            )
        return rbins, vbins, abins, ebins

    def _pre_process(self, adc_samples: np.array) -> tuple[np.array, np.array]:
        """Pre processing of ADC samples.

        The pre-processing step helps in reshaping the data so to match
        the antenna layout of the radar sensor. Some paddings are also
        added if required in order to have a minimum of pre-defined
        frequency bins during FFT processing.

        Since the number of antenna and device configuration can vary
        from one board or recording to another,
        it's good to define a minimum size for the doppler, azimuth, and
        elevation FFT processing (See `config.py` for the default values).

        This does not affect the resolution of the radar sensor but only
        the 3D rendeing.

        Argument:
            adc_samples: Calibrated ADC samples

        Return (tuple):
            virtual_array: 4D data cube ready for FFT processing
            coulpling_calib: Antenna coupling calibration matrix
        """
        virtual_array = rdsp.virtual_array(
            adc_samples,
            self.calibration.antenna.txl,
            self.calibration.antenna.rxl,
        )

        # va_nel: Number of elevations in the virtual array
        # va_naz: Number of azimuth in the virtual array
        # va_nc: Number of chirp per antenna in the virtual array
        # va_ns: Number of samples per chirp
        va_nel, va_naz, va_nc, va_ns = virtual_array.shape

        Ne, Na, Nc, Ns = self._get_fft_size(*virtual_array.shape)

        virtual_array *= np.blackman(va_ns).reshape(1, 1, 1, -1)
        virtual_array = np.pad(
            virtual_array,
            (
                (0, Ne - va_nel), (0, Na - va_naz),
                (0, Nc - va_nc), (0, Ns - va_ns)
            ),
            "constant",
            constant_values=((0, 0), (0, 0), (0, 0), (0, 0))
        )

        coupling_calib = rdsp.virtual_array(
            self.calibration.get_coupling_calibration(),
            self.calibration.antenna.txl,
            self.calibration.antenna.rxl,
        )
        coupling_calib = np.pad(
            coupling_calib,
            (
                (0, Ne - va_nel), (0, Na - va_naz),
                (0, 0), (0, Ns - va_ns)
            ),
            "constant",
            constant_values=((0, 0), (0, 0), (0, 0), (0, 0))
        )
        return virtual_array, coupling_calib


    def _process_raw_adc(self) -> np.array:
        """Radar Signal Processing on raw ADC data.

        FFT Signal processing is applied on the raw Radar ADC samples.
        As a result, we get the Range, Doppler and Angle estimation of
        targets detected by the radar.

        NOTE: The Angle estimation based on FFT doesn't provide high accurary.
        Thus, more advanced methods like MUSIC or ESPRIT could be implemented.
        """
        # Calibrate raw data
        adc_samples = self._calibrate()

        virtual_array, coupling_calib = self._pre_process(adc_samples)

        # Ne: Number of elevations in the virtual array
        # Na: Number of azimuth in the virtual array
        # Nc: Number of chirp per antenna in the virtual array
        # Ns: Number of samples per chirp
        va_ne, va_na, va_nc, va_ns = virtual_array.shape

        Ne, Na, Nc, Ns = self._get_fft_size(*virtual_array.shape)

        # Range-FFT
        virtual_array *= np.blackman(va_ns).reshape(1, 1, 1, -1)
        rfft = np.fft.fft(virtual_array, Ns, -1)
        rfft = rfft - coupling_calib

        # Doppler-FFT
        rfft *= np.blackman(va_nc).reshape(1, 1, -1, 1)
        dfft = np.fft.fft(rfft, Nc, -2)
        dfft = np.fft.fftshift(dfft, -2)
        # dfft = rdsp.velocity_compensation(dfft, ntx, nrx, nc)

        dfft *= np.blackman(va_na).reshape(1, -1, 1, 1)
        # Azimuth estimation
        afft = np.fft.fft(dfft, Na, 1)
        afft = np.fft.fftshift(afft, 1)

        afft *= np.blackman(va_ne).reshape(-1, 1, 1, 1)
        # Elevation esitamtion
        efft = np.fft.fft(afft, Ne, 0)
        efft = np.fft.fftshift(efft, 0)

        # Return the signal power
        return np.abs(efft) ** 2

    def _generate_radar_pcl(self) -> np.array:
        """Generate point cloud."""
        # Calibrate raw data
        adc_samples = self._calibrate()

        # virtual_array, coupling_calib = self._pre_process(adc_samples)

        # ntx: Number of TX antenna
        # nrx: Number of RX antenna
        # nc: Number of chirp per antenna in the virtual array
        # ns: Number of samples per chirp
        ntx, nrx, nc, ns = adc_samples.shape

        rsignal = np.zeros((ntx, nrx, nc, ns), dtype=np.complex64)

        for tidx in range(ntx):
            for ridx in range(nrx):
                samples = adc_samples[tidx, ridx, :, :]
                samples *= np.blackman(ns).reshape(1, -1)
                rfft = np.fft.fft(samples, ns, -1)
                rfft = rfft - self.calibration.get_coupling_calibration()[tidx, ridx, :, :]

                # Doppler-FFT
                rfft *= np.blackman(nc).reshape(-1, 1)
                dfft = np.fft.fft(rfft, nc, -2)
                # dfft = rdsp.velocity_compensation(dfft, ntx, nrx, nc)
                dfft = np.fft.fftshift(dfft, -2)

                rsignal[tidx, ridx, :, :] = dfft

        mimo_dfft = rsignal.reshape(ntx * nrx, nc, ns)
        mimo_dfft = np.sum(np.abs(mimo_dfft) ** 2, 0)

        # OS-CFAR for object detection
        _, detections = rdsp.nq_cfar_2d(mimo_dfft, 8, 2)

        # Restructure data based on the virtual antenna
        va = rdsp.virtual_array(
            rsignal,
            self.calibration.antenna.txl,
            self.calibration.antenna.rxl
        )
        va_ne, va_na, _, _ = va.shape

        # Optional FFT size respectively for the elevation, azimuth,
        # doppler and range axis
        Ne, Na, Nc, Ns = self._get_fft_size(*va.shape)

        # Range, doppler, azimuth and elevation bins
        rbins, vbins, abins, ebins = self._get_bins(Ns, Nc, Na, Ne)

        # Azimuth estimation
        va *= np.blackman(va_na).reshape(1, -1, 1, 1)
        afft = np.fft.fft(va, Na, 1)
        afft = np.fft.fftshift(afft, 1)

        # Elevation esitamtion
        _afft = afft * np.blackman(va_ne).reshape(-1, 1, 1, 1)
        efft = np.fft.fft(_afft, Ne, 0)
        efft = np.fft.fftshift(efft, 0)

        pcl = np.zeros((len(detections), 5))

        for idx, obj in enumerate(detections):
            #Azimuth estimation
            _az = np.argmax(np.sum(afft[:, :, obj.vidx, obj.ridx], 0))
            obj.aidx = _az

            # Elevation estimation
            _el = np.argmax(efft[:, _az, obj.vidx, obj.ridx])
            obj.eidx = _el

            pcl[idx] = np.array([
                abins[obj.aidx],            # Azimnuth
                rbins[obj.ridx],            # Range
                ebins[obj.eidx],            # Elevation
                vbins[obj.vidx],            # Velocity
                10 * np.log10(obj.snr)      # SNR
            ])
        return pcl

    def showPointcloudFromRaw(self,
            velocity_view: bool = False,
            bird_eye_view: bool = False, polar: bool = False, **kwargs) -> None:
        """Render pointcloud of detected object from radar signal processing.

        Arguments:
            bird_eye_view: Enable 2D Bird Eye View rendering
        """
        # ADC sampling frequency
        fs: float = self.calibration.waveform.adc_sample_frequency

        # Frequency slope
        fslope: float = self.calibration.waveform.frequency_slope

        # Maximum range
        rmax: float = rdsp.get_max_range(fs, fslope)

        pcl = self._generate_radar_pcl()
        # Remove very close range
        pcl = pcl[pcl[:, 1] >= 1.5]

        # Exclude all points detected in the last range bins because
        # those detections are not reliable
        pcl = pcl[pcl[:, 1] < (0.95 * rmax)]

        if not polar:
            pcl = self._to_cartesian(pcl)
            pcl = pcl[:, (1, 0, 2, 3, 4)]  # swap range and azimuth
        if bird_eye_view:
            ax = plt.axes()
            ax.set_title("Radar Bird Eye View")
            ax.scatter(
                pcl[:, 0],
                pcl[:, 1],
                pcl[:, 4],
                c=pcl[:, 4],
                cmap="viridis",
            )
            ax.set_xlabel("Azimuth (m)")
            ax.set(facecolor="black")
        else:
            ax = plt.axes(projection="3d")
            ax.set_title("4D-FFT processing of raw ADC samples")
            ax.set_zlabel("Elevation")
            map = ax.scatter(
                pcl[:, 0],
                pcl[:, 1],
                pcl[:, 2],
                c=pcl[:, 3] if velocity_view else pcl[:, 4],
                cmap=plt.cm.get_cmap(),
                s=4.0, # Marker size
            )
            plt.colorbar(map, ax=ax)

        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Range")

        if polar:
            ax.set_xlim(-1, 1)
        else:
            ax.set_xlim(-rmax/2, rmax/2)
        ax.set_ylim(0, rmax)

        if kwargs.get("show", True):
            plt.show()

    def _process_raw_adc_2d(self) -> np.array:
        """Radar Signal Processing on raw ADC data.

        FFT Signal processing is applied on the raw Radar ADC samples.
        As a result, we get the Range, Doppler and Azimuth estimation of
        targets detected by the radar.

        NOTE: The Angle estimation based on FFT doesn't provide high accurary.
        Thus, more advanced methods like MUSIC or ESPRIT should be implemented.
        """
        # Calibrate raw data
        adc_samples = self._calibrate()

        # Only consider azimuth on elevation 0
        virtual_array, coupling_calib = self._pre_process(adc_samples)
        virtual_array = np.sum(virtual_array, 0)
        coupling_calib = np.sum(coupling_calib, 0)

        # Na: Number of azimuth in the virtual array
        # Nc: Number of chirp per antenna in the virtual array
        # Ns: Number of samples per chirp
        va_na, va_nc, va_ns = virtual_array.shape

        _, Na, Nc, Ns = self._get_fft_size(None, va_na, va_nc, va_ns)

        # Range-FFT
        virtual_array *= np.blackman(va_ns).reshape(1, 1, -1)
        rfft = np.fft.fft(virtual_array, Ns, -1)
        rfft = rfft - coupling_calib

        rfft *= np.blackman(va_nc).reshape(1, -1, 1)
        # Doppler-FFT
        dfft = np.fft.fft(rfft, Nc, 1)
        dfft = np.fft.fftshift(dfft, 1)
        # dfft = rdsp.velocity_compensation(dfft, ntx, nrx, nc)

        dfft *= np.blackman(va_na).reshape(-1, 1, 1)
        # Azimuth estimation
        afft = np.fft.fft(dfft, Na, 0)
        afft = np.fft.fftshift(afft, 0)

        # Return the signal power
        return np.abs(afft) ** 2

    def showHeatmapFromRaw(self, threshold: float,
            no_sidelobe: bool = False,
            velocity_view: bool = False,
            polar: bool = False,
            ranges: tuple[Optional[float], Optional[float]] = (None, None),
            azimuths: tuple[Optional[float], Optional[float]] = (None, None),
            **kwargs,
        ) -> None:
        """Render the heatmap processed from the raw radar ADC samples.

        Argument:
            threshold: Threshold value for filtering the heatmap obtained
                       from the radar data processing
            no_sidelobe: Flag used to skip the close range data from rendering
                        in order to avoid side lobes:

                        Consider Range > ~1.5m
            velocity_view: Render the heatmap using the velocity as the fourth
                           dimension. When false, the signal gain in dB is used
                           instead.
            polar (bool): Flag to indicate that the pointcloud should be rendered
                          directly in the polar coordinate. When false, the
                          heatmap is converted into the cartesian coordinate
            ranges (tuple): Min and max value of the range to render
                            Format: (min_range, max_range)
            azimuths (tuple): Min and max value of azimuth to render
                            Format: (min_azimuth, max_azimuth)
            kwargs (dict): Optional keyword arguments
                    "show": When false, prevent the rendered heatmap to be shown
        """
        if self.raw is None:
            info("No raw ADC samples available!")
            return None

        signal_power = self._process_raw_adc()

        # Size of elevation, azimuth, doppler, and range bins
        Ne, Na, Nv, Nr = signal_power.shape
        # Range, Doppler, Azimuth and Elevation bins
        rbins, vbins, abins, ebins = self._get_bins(Nr, Nv, Na, Ne)

        #
        # Noise filering masks
        #

        # Doppler Non-coherent integration
        sp = np.sum(signal_power, (0, 1, 3))
        dmask= rdsp.os_cfar(
            sp,
            self.CFAR_WS//2,
            self.CFAR_GC//2
        ).reshape(1, 1, Nv, 1)

        # Non-coherent integration with Azimuth-Range 2D noise filtering
        sp = np.sum(signal_power, (0, 2))
        rmask, _ = rdsp.nq_cfar_2d(
            sp,
            self.CFAR_WS,
            self.CFAR_GC
        )
        rmask = rmask.reshape(1, Na, 1, Nr)

        dpcl = 10 * np.log10(signal_power + 1)
        dpcl *= dmask * rmask
        dpcl /= np.max(dpcl)

        hmap = np.zeros((Ne * Na * Nv * Nr, 5))

        # Range offset to count for the side lobes of the radar self.sensor
        roffset: int = 10

        if no_sidelobe:
            dpcl[:, :, :, 0:roffset] = 0.0

        '''
        NOTE
        -----
        The time complexity of the nested loop below is in the order of:
            Ne * Na* * Nv * ns (about N^4)
        Hence it takes eons when higher resolution of rendering is used.
        This motivated the vectorized implementation which is ~25 times faster.

        for elidx in range(Ne):
            for aidx in range(Na):
                for vidx in range(Nv):
                    for ridx in range(roffset, ns):
                        hmap_idx: int = ridx + ns * (
                            vidx + Nv * ( aidx + Na * elidx)
                        )
                        hmap[hmap_idx] = np.array([
                            abins[aidx],
                            rbins[ridx],
                            ebins[elidx],
                            vbins[vidx],
                            dpcl[elidx, aidx, vidx, ridx],
                        ])
        '''
        hmap[:, 0] = np.repeat(
            [np.repeat(abins, Nv * Nr)],
            Ne, axis=0
        ).reshape(-1)
        hmap[:, 1] = np.repeat(
            [np.repeat([np.repeat([rbins], Nv, axis=0)], Na, axis=0)],
            Ne, axis=0
        ).reshape(-1)
        hmap[:, 2] = np.repeat(ebins, Na * Nv * Nr)
        hmap[:, 3] = np.repeat(
            [np.repeat([np.repeat(vbins, Nr)], Na, axis=0)],
            Ne, axis=0
        ).reshape(-1)
        hmap[:, 4] = dpcl.reshape(-1)

        # Filtering ranges
        min_range, max_range = ranges
        if min_range is not None:
            hmap = hmap[hmap[:, 1] > min_range]
        if max_range is not None:
            hmap = hmap[hmap[:, 1] < max_range]

        # Filtering azimuths
        min_az, max_az= azimuths
        if min_az is not None:
            hmap = hmap[hmap[:, 0] > min_az]
        if max_az is not None:
            hmap = hmap[hmap[:, 0] < max_az]

        hmap = hmap[hmap[:, 4] > threshold]
        # Re-Normalise the radar reflection intensity after filtering
        hmap[:, 4] -= np.abs(np.quantile(hmap[:, 4], 0.96, method='weibull'))
        hmap = hmap[hmap[:, 4] >  0]
        hmap[:, 4] -= np.min(hmap[:, 4])
        hmap[:, 4] /= np.max(hmap[:, 4])

        if not polar:
            hmap = self._to_cartesian(hmap)
            # Swap range and azimuth axis
            hmap = hmap[:, (1, 0, 2, 3, 4)]

        ax = plt.axes(projection="3d")
        ax.set_title("4D-FFT processing of raw ADC samples")
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Range")
        ax.set_zlabel("Elevation")
        if polar:
            ax.set_xlim(np.min(abins), np.max(abins))
        else:
            ax.set_xlim(-np.max(rbins), np.max(rbins))
        ax.set_ylim(np.min(rbins), np.max(rbins))
        ax.view_init(azim=-30, elev=45)
        map = ax.scatter(
            hmap[:, 0],
            hmap[:, 1],
            hmap[:, 2],
            c=hmap[:, 3] if velocity_view else hmap[:, 4],
            cmap=plt.cm.get_cmap(),
            s=5.0, # Marker size
        )
        plt.colorbar(map, ax=ax)
        if kwargs.get("show", True):
            plt.show()

    def show2dHeatmap(self, polar: bool = False, show: bool = True) -> None:
        """Render 2D heatmap.

        Argument:
            polar: Flag to indicate that the pointcloud should be rendered
                   directly in the polar coordinate. When false, the
                   heatmap is converted into the cartesian coordinate
            show: Enable the rendered figure to be shown.
        """
        if self.raw is None:
            info("No raw ADC samples available!")
            return None

        signal_power = self._process_raw_adc_2d()

        # Size of elevation, azimuth, doppler, and range bins
        Na, Nv, Nr = signal_power.shape
        rbins, _, abins, _ = self._get_bins(Nr, None, Na, None)

        # Noise filtering mask
        sp = np.sum(signal_power, 1)
        mask, _ = rdsp.nq_cfar_2d(sp, self.CFAR_WS, self.CFAR_GC)

        dpcl = np.log10(signal_power)
        dpcl = np.sum(dpcl, 1)
        dpcl *= mask
        dpcl /= np.max(dpcl)

        # Number of close range bins to skip
        roffset: int = 15

        if not polar:
            hmap = np.zeros((Na * Nr, 3))

            for aidx, _az in enumerate(abins):
                for ridx, _r in enumerate(rbins[roffset:]):
                    hmap[ridx + roffset + Nr * aidx] = np.array([
                        _r * np.sin(_az),   # Azimuth
                        _r * np.cos(_az),   # Range
                        dpcl[aidx, ridx + roffset],
                    ])

            ax = plt.axes()
            ax.scatter(
                hmap[:, 0],
                hmap[:, 1],
                hmap[:, 2],
                c=hmap[:, 2],
            )
            ax.set_xlabel("Azimuth (m)")
            ax.set(facecolor="black")
        else:
            dpcl = np.transpose(dpcl, (1, 0))
            az, rg = np.meshgrid(abins, rbins)
            _, ax = plt.subplots()
            color = ax.pcolormesh(az, rg, dpcl, cmap="viridis")
            ax.set_xlabel("Azimuth (rad)")

        ax.set_ylabel("Range (m)")
        ax.set_title(f"Frame {self.index:04}")
        if show:
            plt.show()


class CCRadar(SCRadar):
    """Cascade Chip Radar.

    Attributes:
        CFAR_WS: Constant False Alarm Rate Window Size
        CFAR_GC: Constant False Alarm Rate Guard Cell
    """

    CFAR_WS: int = 32
    CFAR_GC: int = 16
    AZIMUTH_FOV: float = np.deg2rad(90)
    ELEVATION_FOV: float = np.deg2rad(20)


    def _phase_calibration(self) -> np.array:
        """Apply Phase calibration.

        Return:
            Phase calibrated ADC samples
        """
        # Phase calibrationm atrix
        pm = np.array(self.calibration.phase.phase_calibration_matrix)
        pm = pm[0] / pm
        phase_calibrated_mtx = self.raw * pm.reshape(
            self.calibration.phase.num_tx,
            self.calibration.phase.num_rx,
            1,
            1
        )
        return phase_calibrated_mtx
