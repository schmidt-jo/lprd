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


def compress_channels(input_k_space: np.ndarray, num_compressed_channels: int = 8):
    """ k-space is assumed to be provided in dims [x, y, z, ch, t (optional)]"""
    if input_k_space.shape.__len__() < 5:
        input_k_space = np.expand_dims(input_k_space, -1)
    # check if we are actually provided fewer channels
    num_in_ch = input_k_space.shape[-2]
    if num_in_ch < num_compressed_channels:
        err = f"input data has fewer channels ({num_in_ch}) than set for compression ({num_compressed_channels})!"
        log_module.error(err)
        raise ValueError(err)
    log_module.debug(f"start pca -> building compression matrix from data")
    # we dont want to take this across echoes
    compressed_data = np.zeros_like(input_k_space[:, :, :, :num_compressed_channels])
    for idx_echo in range(input_k_space.shape[-1]):
        pca_data = input_k_space[:, :, :, :, idx_echo]

        # set channel dimension first and rearrange rest
        pca_data = np.moveaxis(pca_data, -1, 0)
        pca_data = np.reshape(pca_data, (num_in_ch, -1))
        # substract mean from each channel vector
        pca_data = pca_data - np.mean(pca_data, axis=1, keepdims=True)
        # calculate covariance matrix
        cov = np.cov(pca_data)
        cov_eig_val, cov_eig_vec = np.linalg.eig(cov)
        # get coil compression matrix
        a_l_matrix = cov_eig_vec[:num_compressed_channels]
        log_module.debug(f"compressing data channels from {num_in_ch} to {num_compressed_channels}")

        # compress data -> coil dimension over a_l
        compressed_data[:, :, :, :, idx_echo] = np.squeeze(
            np.einsum("xyzo, co -> xyzc", input_k_space[:, :, :, :, idx_echo], a_l_matrix)
        )
    return compressed_data
