"""Radar.

SCRadar: Single Chip Radar Sensor
CCRadar: Cascade Chip Radar Sensor
"""
import os
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt

from core.calibration import Calibration, SCRadarCalibration
from core.utils.common import error
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
    """

    # The recorded attributes are:
    # x, y, z, I (Intensity of the reflections), Vr (Radial velocity)
    NUMBER_RECORDING_ATTRIBUTES: int = 5

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

        if self.sensor == "scradar":
            # Read pointcloud
            filename: str = self._filename(
                config["paths"][self.sensor]["pointcloud"]["filename_prefix"],
                index,
                "bin"
            )
            self.filepath = os.path.join(
                config["paths"]["rootdir"],
                config["paths"][self.sensor]["pointcloud"]["data"],
                filename
            )
            try:
                cld = np.fromfile(self.filepath, np.float32)
                self.cld = np.reshape(cld, (-1, self.NUMBER_RECORDING_ATTRIBUTES))
            except FileNotFoundError:
                error(f"File '{self.filepath}' not found.")
                sys.exit(1)

        # Read heatmap
        filename = self._filename(
            config["paths"][self.sensor]["heatmap"]["filename_prefix"],
            index,
            "bin"
        )
        self.heatmap_filepath = os.path.join(
            config["paths"]["rootdir"],
            config["paths"][self.sensor]["heatmap"]["data"],
            filename
        )
        try:
            heatmap = np.fromfile(self.heatmap_filepath, np.float32)
            self.heatmap = np.reshape(
                heatmap,
                (
                    self.calibration.heatmap.num_elevation_bins,
                    self.calibration.heatmap.num_azimuth_bins,
                    self.calibration.heatmap.num_range_bins,
                    2 # Number of value per bin (intensity and location)
                )
            )
        except FileNotFoundError:
            error(f"File '{self.heatmap_filepath}' not found.")
            sys.exit(1)

        # read raw ADC measurements
        filename = self._filename(
            config["paths"][self.sensor]["raw"]["filename_prefix"],
            index,
            "bin"
        )
        self.raw_meas_filepath = os.path.join(
            config["paths"]["rootdir"],
            config["paths"][self.sensor]["raw"]["data"],
            filename
        )
        try:
            raw = np.fromfile(self.raw_meas_filepath, np.int16)
            # I measurements: raw[::2]
            # Q measurements: raw[1::2]
            # s = I + jQ
            raw = np.float16(raw[::2]) + 1j * np.float16(raw[1::2])
            self.raw = np.reshape(
                raw,
                (
                    self.calibration.waveform.num_tx,
                    self.calibration.waveform.num_rx,
                    self.calibration.waveform.num_chirps_per_frame,
                    self.calibration.waveform.num_adc_samples_per_chirp,
                ),
            )
        except FileNotFoundError:
            error(f"File '{self.raw_meas_filepath}' not found.")
            sys.exit(1)

    def showHeatmap(self, threshold: float = 0.15, no_sidelobe: bool = False,
                          render: bool = True) -> None:
        """Render heatmap.

        Argument:
            threshold: Value used to filter the pointcloud
            no_sidelobe: Flag used to skip the close range data from rendering
                         in order to avoid side lobes
            render: Flag triggering the rendering of the heatmap when 'true'.
        """
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

    def _process_raw_adc(self) -> np.array:
        """Radar Signal Processing on raw ADC data.

        FFT Signal processing is applied on the raw Radar ADC samples.
        As a result, we get the Range, Doppler and Angle estimation of
        targets detected by the radar.

        Since the number of antenna and device configuration can vary
        from one board or recording to another,
        it's good to define a minimum size for the doppler, azimuth, and
        elevation FFT processing. Let's consider

            (Azimuth) Na: 32 
            (Elevation) Ne: 16

        This does not affect the resolution of the radar sensor but only
        the 3D rendeing.


        A minimum elevation and azimuth bin size 

        NOTE: The Angle estimation based on FFT doesn't provide high accurary.
        Thus, more advanced methods like MUSIC or ESPRIT should be implemented.
        """
        # Calibrate raw data
        adc_samples = self.raw

        if self.sensor != "scradar":
            adc_samples *= self.calibration.get_frequency_calibration()
            adc_samples *= self.calibration.get_phase_calibration()

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

        # Estimated size of the elevation and azimuth
        Ne = rdsp.fft_size(va_nel)
        Na = rdsp.fft_size(va_naz)
        Na = Na if Na > NUMBER_AZIMUTH_BINS_MIN else NUMBER_AZIMUTH_BINS_MIN
        Ne = (
            Ne if Ne > NUMBER_ELEVATION_BINS_MIN else NUMBER_ELEVATION_BINS_MIN
        )

        # Size of doppler FFT
        Nc = rdsp.fft_size(va_nc)
        Nc = Nc if Nc > NUMBER_DOPPLER_BINS_MIN else NUMBER_DOPPLER_BINS_MIN

        # Size of range FFT
        Ns = rdsp.fft_size(va_ns)
        Ns = Ns if Ns > NUMBER_RANGE_BINS_MIN else NUMBER_RANGE_BINS_MIN

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

        # Range-FFT
        # adc_samples *= np.blackman(ns).reshape(1, 1, -1)
        rfft = np.fft.fft(virtual_array, Ns, -1)
        rfft = rfft - coupling_calib

        # Doppler-FFT
        dfft = np.fft.fft(rfft, Nc, -2)
        # dfft = rdsp.velocity_compensation(dfft, ntx, nrx, nc)

        # Azimuth estimation
        afft = np.fft.fft(dfft, Na, 1)

        # Elevation esitamtion
        afft = np.fft.fft(afft, Ne, 0)

        afft = np.fft.fftshift(afft, (0, 1, 2))

        # Return the signal power
        return np.abs(afft) ** 2


    def showHeatmapFromRaw(self, threshold: float,
            no_sidelobe: bool = False, velocity_view: bool = False) -> None:
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
        """
        stime = time()

        ntx: int = self.calibration.waveform.num_tx

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

        signal_power = self._process_raw_adc()

        print(f"Processing end time: {time() - stime} s")

        # Size of elevation, azimuth, doppler, and range bins
        Ne, Na, Nv, Nr = signal_power.shape

        # Range bins
        rbins = rdsp.get_range_bins(Nr, fs, fslope)
        # Velocity bins
        vbins = rdsp.get_velocity_bins(ntx, Nv, fstart, tc)
        # Azimuth bins
        ares = np.pi / Na
        abins = np.arange(-np.pi/2, np.pi/2, ares)
        # abins = np.flip(abins)
        # Elevation
        eres = np.pi / Ne
        ebins = np.arange(-np.pi/2, np.pi/2, eres)

        dpcl = 10 * np.log10(signal_power + 1)
        dpcl -= np.quantile(dpcl, 0.98, method='weibull')
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
        stime = time()
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

        print(f"Heatmap building time: {time() - stime} s")

        hmap = hmap[hmap[:, 4] > threshold]
        # Re-Normalise the radar reflection intensity after filtering
        hmap[:, 4] -= np.quantile(hmap[:, 4], 0.96, method='weibull')
        hmap = hmap[hmap[:, 4] >  0]
        hmap[:, 4] -= np.min(hmap[:, 4])
        hmap[:, 4] /= np.max(hmap[:, 4])

        ax = plt.axes(projection="3d")
        ax.set_title("4D-FFT processing of raw ADC samples")
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Range")
        ax.set_zlabel("Elevation")
        map = ax.scatter(
            hmap[:, 0],
            hmap[:, 1],
            hmap[:, 2],
            c=hmap[:, 3] if velocity_view else hmap[:, 4],
            cmap=plt.cm.get_cmap()
        )
        plt.colorbar(map, ax=ax)
        plt.show()


class CCRadar(SCRadar):
    """Cascade Chip Radar."""


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
