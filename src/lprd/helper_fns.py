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


def compress_channels(input_k_space: np.ndarray, sampling_pattern: np.ndarray = None, num_compressed_channels: int = 8):
    """ k-space is assumed to be provided in dims [x, y, z, ch, t (optional)]"""
    if input_k_space.shape.__len__() < 5:
        input_k_space = np.expand_dims(input_k_space, -1)
    # check if we are actually provided fewer channels
    num_in_ch = input_k_space.shape[-2]
    if num_in_ch < num_compressed_channels:
        err = f"input data has fewer channels ({num_in_ch}) than set for compression ({num_compressed_channels})!"
        log_module.error(err)
        raise ValueError(err)
    log_module.debug(f"extract ac region from sampling mask")

    if sampling_pattern is None:
        # just use whole image
        sampling_pattern = np.ones(input_k_space.shape[:3], dtype=bool)
    # find ac data - this is implemented for the commonly used (in jstmc) sampling scheme of acquiring middle pe lines
    # use 0th echo, ac should be equal for all slices, all channels and all echoes
    # start search in middle of pe dim
    mid_read, mid_pe, _ = (np.array(sampling_pattern.shape) / 2).astype(int)
    lower_edge = mid_pe
    upper_edge = mid_pe
    # make sure sampling pattern is bool
    sampling_pattern = sampling_pattern.astype(bool)
    for idx_pe in np.arange(1, mid_pe):
        # looking for sampled line (actually just looking for the read middle point)
        # if previous line was also sampled were still in a fully sampled region
        next_up_line_sampled = sampling_pattern[mid_read, mid_pe + idx_pe, 0]
        next_low_line_sampled = sampling_pattern[mid_read, mid_pe - idx_pe, 0]
        prev_up_line_sampled = sampling_pattern[mid_read, mid_pe + idx_pe - 1, 0]
        prev_low_line_sampled = sampling_pattern[mid_read, mid_pe - idx_pe + 1, 0]
        if next_up_line_sampled and prev_up_line_sampled:
            upper_edge = mid_pe + idx_pe
        if next_low_line_sampled and prev_low_line_sampled:
            lower_edge = mid_pe - idx_pe
        if not next_up_line_sampled or not next_low_line_sampled:
            # get out if we hit first unsampled line
            break
    # extract ac region
    ac_mask = np.zeros_like(sampling_pattern[:, :, 0])
    ac_mask[:, lower_edge:upper_edge] = True
    log_module.debug(f"start pca -> building compression matrix from calibration data")
    # set input data
    pca_data = input_k_space[ac_mask]
    # set channel dimension first and rearrange rest
    pca_data = np.moveaxis(pca_data, -2, 0)
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
    compressed_data = np.einsum("iklmn, om -> iklon", input_k_space, a_l_matrix)
    return compressed_data

