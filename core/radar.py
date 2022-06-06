"""Radar.

SCRadar: Single Chip Radar Sensor
CCRadar: Cascade Chip Radar Sensor
"""
from hmac import digest_size
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from core.calibration import Calibration, SCRadarCalibration
from core.utils.common import error
from core.utils import radardsp as rdsp
from .lidar import Lidar
from .config import DEBUG


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
        sensor: str = self.__class__.__name__.lower()
        self.calibration: SCRadarCalibration = getattr(calib, sensor)

        if sensor == "scradar":
            # Read pointcloud
            filename: str = self._filename(
                config["paths"][sensor]["pointcloud"]["filename_prefix"],
                index,
                "bin"
            )
            self.filepath = os.path.join(
                config["paths"]["rootdir"],
                config["paths"][sensor]["pointcloud"]["data"],
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
            config["paths"][sensor]["heatmap"]["filename_prefix"],
            index,
            "bin"
        )
        self.heatmap_filepath = os.path.join(
            config["paths"]["rootdir"],
            config["paths"][sensor]["heatmap"]["data"],
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
            config["paths"][sensor]["raw"]["filename_prefix"],
            index,
            "bin"
        )
        self.raw_meas_filepath = os.path.join(
            config["paths"]["rootdir"],
            config["paths"][sensor]["raw"]["data"],
            filename
        )
        if os.path.exists(self.raw_meas_filepath):
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
        elif DEBUG:
            error(f"File '{self.raw_meas_filepath}' not found.")

    def showHeatmap(self, threshold: float = 0.15, render: bool = True) -> None:
        """Render heatmap.

        Argument:
            threshold: Value used to filter the pointcloud
            render: Flag triggering the rendering of the heatmap when 'true'.
        """
        ax = plt.axes(projection="3d")
        ax.set_title("Heatmap")
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Range")
        ax.set_zlabel("Elevation")
        pcl = self._heatmap_to_pointcloud(threshold)
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

    def _heatmap_to_pointcloud(self, threshold: float = 0.15) -> np.array:
        """Compute pointcloud from heatmap.

        The recordings of the heatmap are stored in the polar form.
        This function prepares a pointcloud in the cartesian coordinate
        system based the heatmap

        Argument:
            threshold (float): Threshold to filter the pointcloud

        Return
            pcl (np.array): the heatmap point locations
        """
        _num_el_bin: int = self.calibration.heatmap.num_elevation_bins
        _num_az_bin: int = self.calibration.heatmap.num_azimuth_bins
        _num_r_num: int = self.calibration.heatmap.num_range_bins

        _range_bin_width: float = self.calibration.heatmap.range_bin_width

        # transform range-azimuth-elevation heatmap to pointcloud
        pcl = np.zeros((_num_el_bin, _num_az_bin, _num_r_num, 5))

        for range_idx in range(_num_r_num - 1):
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

    def _rx_antenna_layout(self) -> np.array:
        """Return the layout of the RX antenna.

        The spacing of the antenna is indicated in unit of 1/2 of wavelength

            (rx0) <-- 1/2 lambda --> (rx1) <-- 1/2 lambda --> (rx2 ...)
        """
        return np.array([
            0, 1, 2, 3,         # WR1
        ])

    def _az_tx_antenna_layout(self) -> np.array:
        """Return the azimuth layout of the TX antenna.

        The spacing of the antenna is indicated in unit of 1/2 of wavelength
        """
        return np.array([
            0, 4
        ])

    def _el_tx_antenna_layout(self) -> np.array:
        """Return the elevation layout of the TX antenna.

        The spacing of the antenna is indicated in unit of 1/2 of wavelength
        """
        return np.array([
           0, 2
        ])

    def _process_raw_adc(self):
        """Radar Signal Processing on raw ADC data.

        FFT Signal processing is applied on the raw Radar ADC samples.
        As a result, we get the range doppler and angle estimation of
        targets detected by the radar.

        NOTE: The Angle estimation based on FFT is not accurate and more
        advanced methods like MUSIC or ESPRIT should be implemented.
        """

        # Calibrate raw data
        adc_samples = self.raw

        # self._format_adc(adc_samples)

        ntx: int = self.calibration.waveform.num_tx
        nrx: int = self.calibration.waveform.num_rx
        nc: int = self.calibration.waveform.num_chirps_per_frame
        ns: int = self.calibration.waveform.num_adc_samples_per_chirp

        # ADC sampling frequency
        fs: float = self.calibration.waveform.adc_sample_frequency

        # Frequency slope
        fslope: float = self.calibration.waveform.frequency_slope

        # Chrip start frequency
        fstart: float = self.calibration.waveform.start_frequency

        # Chirp time
        tc: float = self.calibration.waveform.idle_time + self.calibration.waveform.ramp_end_time

        # Range-FFT
        rfft = rdsp.range_fft(adc_samples, ns, self.calibration.get_coupling_calibration())
        rfft = rfft.reshape(ntx * nrx, nc, -1) * np.blackman(nc).reshape(1, nc)
        # Doppler-FFT
        dfft = rdsp.doppler_fft(rfft, ntx, nrx, nc)


        rncint = rdsp.range_nci(dfft)
        mask = rdsp.os_cfar(rncint.reshape(-1), 16)
        rncint *= mask.reshape(ntx * nrx, -1)

        dfft = dfft * mask.reshape(ntx * nrx, 1, -1)

        vncint = rdsp.doppler_nci(dfft)
        mask = rdsp.os_cfar(vncint.reshape(-1), 16)
        vncint *= mask.reshape(ntx * nrx, -1)

        dfft = dfft * mask.reshape(ntx * nrx, nc, -1)

        dfft = rdsp.velocity_compensation(dfft, ntx, nrx, nc)
        dfft = dfft.reshape(ntx * nrx, nc, -1)
        dfft = np.fft.fftshift(dfft, 1)

        rres = rdsp.get_range_resolution(ns, fs, fslope)
        vres = rdsp.get_velocity_resolution(nc, fstart, tc)

        rbins = rdsp.get_range_bins(ns, fs, fslope)
        # vbins = rdsp.get_velocity_bins(ntx, nrx, nc, fstart, tc)

        # Size of azimuth FFT
        Na: int = 64
        # Size of elevation FFT
        Ne: int = 16

        # Zero filled virtual array
        dfft = np.sum(dfft, 1)
        dfft = dfft.reshape(ntx, nrx, ns)
        virtual_array = np.zeros((2, 8, ns), dtype=np.complex128)
        virtual_array[1, :, :] = dfft[(0, 2), :].reshape(2 * nrx, ns)
        r1 = dfft[1, :, :].reshape(nrx, ns)
        virtual_array[0, 2:(2 + nrx), :] = r1

        virtual_array = np.pad(
            virtual_array,
            ((0, 14), (0, 56), (0, 0)),
            "constant",
            constant_values=((0, 0), (0, 0), (0, 0))
        )

        # Azimuth estimation
        afft = np.fft.fft(virtual_array, Na, 1)
        afft = np.fft.fftshift(afft, 1)
        # Elevation esitamtion
        afft = np.fft.fft(afft, Ne, 0)
        afft = np.fft.fftshift(afft, 0)
        ares = np.pi / Na
        abins = np.arange(-np.pi/2, np.pi/2, ares)
        eres = np.pi / Ne
        ebins = np.arange(-np.pi/2, np.pi/2, eres)

        dpcl = afft

        dpcl = 20 * np.log10(np.abs(dpcl) + 1)
        dpcl -= np.min(dpcl)
        dpcl /= np.max(dpcl)

        hmap = np.zeros((Ne * Na * ns, 4))

        for elidx in range(Ne):
            for aidx in range(Na):
                # for vidx in range(nc):
                for ridx in range(ns):
                    hmap[ridx + ns * ( aidx + Na * elidx)] = np.array(
                        [rbins[ridx], abins[aidx], ebins[elidx], dpcl[elidx, aidx, ridx]]
                    )

        hmap = np.delete(hmap, np.argwhere(hmap[:, 3] == 0.0), axis=0).reshape(-1, 4)

        figure = plt.figure()
        ax = figure.add_subplot(projection="3d")
        map = ax.scatter(hmap[:, 0], hmap[:, 1], hmap[:, 2], c=hmap[:, 3], cmap=plt.cm.get_cmap())
        plt.colorbar(map, ax=ax)
        plt.show()

    def showRaw(self):
        """Render processed raw radar ADC samples."""
        self._process_raw_adc()


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

    def _rx_antenna_layout(self) -> np.array:
        """Return the layout of the RX antenna.

        The spacing of the antenna is indicated in unit of 1/2 of wavelength

            (rx0) <-- 1/2 lambda --> (rx1) <-- 1/2 lambda --> (rx2 ...)
        """
        return np.array([
            0, 1, 2, 3,         # WR4
            11, 12, 13, 14,     # WR1
            46, 47, 48, 49,     # WR3
            50, 51, 52, 53      # WR2
        ])

    def _az_tx_antenna_layout(self) -> np.array:
        """Return the azimuth layout of the TX antenna.

        The spacing of the antenna is indicated in unit of 1/2 of wavelength
        """
        return np.array([
            0, 4, 8,        # WR4
            12, 16, 20,     # WR3
            24, 28, 32      # WR2
        ])

    def _el_tx_antenna_layout(self) -> np.array:
        """Return the elevation layout of the TX antenna.

        The spacing of the antenna is indicated in unit of 1/2 of wavelength
        """
        return np.array([
           0, 1, 4, 6          # WR1
        ])

    def _process_raw_adc(self):
        """Process raw radar ADC samples."""
        # Calibrate raw data
        adc_samples = self.raw
        adc_samples *= self.calibration.get_frequency_calibration()
        adc_samples *= self.calibration.get_phase_calibration()

        ntx: int = self.calibration.waveform.num_tx
        nrx: int = self.calibration.waveform.num_rx
        nc: int = self.calibration.waveform.num_chirps_per_frame
        ns: int = self.calibration.waveform.num_adc_samples_per_chirp

        # ADC sampling frequency
        fs: float = self.calibration.waveform.adc_sample_frequency

        # Frequency slope
        fslope: float = self.calibration.waveform.frequency_slope

        # Chrip start frequency
        fstart: float = self.calibration.waveform.start_frequency

        # Chirp time
        tc: float = self.calibration.waveform.idle_time + self.calibration.waveform.ramp_end_time

        # Range-FFT
        rfft = rdsp.range_fft(adc_samples, ns, self.calibration.get_coupling_calibration())

        rfft *= np.blackman(nc).reshape(1, 1, -1, 1)
        # Doppler-FFT
        dfft = rdsp.doppler_fft(rfft, ntx, nrx, nc)

        rncint = rdsp.range_nci(dfft)
        mask = rdsp.os_cfar(rncint.reshape(-1), 16)
        rncint *= mask.reshape(ntx * nrx, -1)
        dfft = dfft * mask.reshape(ntx * nrx, 1, -1)

        dfft = rdsp.velocity_compensation(dfft, ntx, nrx, nc)
        dfft = np.fft.fftshift(dfft, 1)
        dfft = np.sum(dfft, 1).reshape(ntx, nrx, ns)


        virtual_array = np.zeros((7, 144, ns), dtype=np.complex128)
        virtual_array[0, :, :] = dfft[3:, :, :].reshape(-1, ns)
        virtual_array[1, :16, :] = dfft[0, :, :].reshape(-1, ns)
        virtual_array[4, :16, :] = dfft[1, :, :].reshape(-1, ns)
        virtual_array[6, :16, :] = dfft[2, :, :].reshape(-1, ns)

        virtual_array = np.pad(
            virtual_array,
            ((0, 9), (0, 0), (0, 0)),
            "constant",
            constant_values=((0, 0), (0, 0), (0, 0))
        )

        # Size of azimuth FFT
        Na: int = 64
        # Size of elevation FFT
        Ne: int = 16


        # Azimuth estimation
        afft = np.fft.fft(virtual_array, Na, 1)
        afft = np.fft.fftshift(afft, 1)

        rbins = rdsp.get_range_bins(ns, fs, fslope)
        # vbins = rdsp.get_velocity_bins(ntx, nrx, nc, fstart, tc * nc)

        # Elevation esitamtion
        afft = np.fft.fft(afft, Ne, 0)
        afft = np.fft.fftshift(afft, 0)
        ares = np.pi / Na
        abins = np.arange(-np.pi/2, np.pi/2, ares)
        eres = np.pi / Ne
        ebins = np.arange(-np.pi/2, np.pi/2, eres)

        dpcl = afft

        dpcl = 20 * np.log10(np.abs(dpcl) + 1)
        dpcl -= np.min(dpcl)
        dpcl /= np.max(dpcl)

        hmap = np.zeros((Ne * Na * ns, 4))

        for elidx in range(Ne):
            for aidx in range(Na):
                # for vidx in range(nc):
                for ridx in range(ns):
                    hmap[ridx + ns * ( aidx + Na * elidx)] = np.array(
                        [rbins[ridx], abins[aidx], ebins[elidx], dpcl[elidx, aidx, ridx]]
                    )

        hmap = np.delete(hmap, np.argwhere(hmap[:, 3] == 0), axis=0).reshape(-1, 4)

        figure = plt.figure()
        ax = figure.add_subplot(projection="3d")
        map = ax.scatter(hmap[:, 0], hmap[:, 1], hmap[:, 2], c=hmap[:, 3], cmap=plt.cm.get_cmap())
        plt.colorbar(map, ax=ax)
        plt.show()
