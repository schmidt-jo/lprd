import numpy as np
import logging
import typing

log_module = logging.getLogger(__name__)


def data_fft_to_freq(data: np.ndarray, data_in_time_domain: bool = True, axes: typing.Union[int, tuple] = -1):
    if isinstance(axes, int):
        # make it a tuple
        axes = (axes,)
    if not data_in_time_domain:
        return data, False
    else:
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data, axes=axes), axes=axes), axes=axes), False


def data_fft_to_time(data: np.ndarray, data_in_time_domain: bool = False, axes: typing.Union[int, tuple] = -1):
    if isinstance(axes, int):
        # make it a tuple
        axes = (axes,)
    if data_in_time_domain:
        return data, True
    else:
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data, axes=axes), axes=axes), axes=axes), True


def remove_os(data: np.ndarray, data_input_sampled_in_time: bool = True, read_dir: int = 0, os_factor: int = 2):
    log_module.info(f"remove oversampling")
    nx = data.shape[read_dir]

    data, data_processing_in_time_domain = data_fft_to_freq(
        data=data, data_in_time_domain=data_input_sampled_in_time, axes=read_dir
    )
    lower_idx = int((os_factor - 1) / (2 * os_factor) * nx)
    upper_idx = int((os_factor + 1) / (2 * os_factor) * nx)
    data = np.moveaxis(data, read_dir, 0)[lower_idx:upper_idx]
    data = np.moveaxis(data, 0, read_dir)
    if data_input_sampled_in_time:
        data, data_processing_in_time_domain = data_fft_to_time(
            data=data, data_in_time_domain=data_processing_in_time_domain, axes=read_dir
        )
    return data


def apply_phase_wrap(phase_data: np.ndarray):
    return np.angle(
        np.exp(1j * phase_data)
    )
