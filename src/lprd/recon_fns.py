import numpy as np
import scipy.ndimage as sndi
import logging
from lprd import helper_fns
import tqdm

log_module = logging.getLogger(__name__)


def fft_mag_sos_phase_aspire_recon(k_data_xyz_ch_t: np.ndarray,
                                   se_indices: tuple = (0, 1),
                                   combine_phase: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    # fft
    data_xyz_ch_t, _ = helper_fns.data_fft_to_time(data=k_data_xyz_ch_t, axes=(0, 1))
    # try compression before combination
    num_compress_channels = 8
    data_xyz_ch_t_compressed = np.zeros(
        (*data_xyz_ch_t.shape[:3], num_compress_channels, data_xyz_ch_t.shape[-1])
    )
    for echo_idx in tqdm.trange(k_data_xyz_ch_t.shape[-1], desc="compress channels per echo"):
        data_xyz_ch_t_compressed[:, :, :, :, echo_idx] = np.squeeze(
            helper_fns.compress_channels(
                input_k_space=data_xyz_ch_t[:, :, :, :, echo_idx], num_compressed_channels=num_compress_channels
            )
        )

    data_xyzt_mag = magnitude_coil_combination(data_xyz_ch_t=data_xyz_ch_t_compressed)

    if combine_phase:
        data_xyzt_ph, grad_offset = phase_coil_combination(data_xyz_ch_t=data_xyz_ch_t, se_indices=se_indices)
    else:
        data_xyzt_ph = np.ones_like(data_xyzt_mag)
        grad_offset = np.ones_like(data_xyzt_mag)
    return data_xyzt_mag, data_xyzt_ph, grad_offset


def magnitude_coil_combination(data_xyz_ch_t: np.ndarray) -> np.ndarray:
    log_module.info(f"processing magnitude coil combination")
    # magnitude sos - assume channel in 3 dim - possibly complex data! we need to use rSoS on magnitude data
    return np.abs(
        np.sqrt(
            np.sum(
                np.square(
                    np.abs(data_xyz_ch_t)
                ),
                axis=3
            )
        )
    )


def phase_coil_combination(data_xyz_ch_t: np.ndarray, se_indices: tuple = (0, 1)) -> (np.ndarray, np.ndarray):
    """
    phase coil combination following ASPIRE, Eckstein et al. doi: 10.1002/mrm.26963

    :param data_xyz_ch_t: img space data - xyz-ch-t
    :param se_indices: indices of spin echoes in time dimension, usually (0,1) for cpmg, (2,5) for megesse
    :return:
    """
    log_module.info(f"processing phase coil combination")
    log_module.info(f"\t\tset echo images to use: {se_indices}")
    # sanity check
    if len(data_xyz_ch_t.shape) != 5:
        err = f"provided input data (shape {data_xyz_ch_t.shape}) does not match required shape (x,y,z,ch,t)"
        log_module.exception(err)
        raise AttributeError(err)
    log_module.debug(f"build magnitude and phase data")
    m_j = np.abs(data_xyz_ch_t)
    phi_j = np.angle(data_xyz_ch_t)
    log_module.debug(f"filter first echo magnitude data for approximate coil sensitivity weighting")
    m_weights = sndi.gaussian_filter(m_j[:, :, :, :, 0], sigma=10, axes=(0, 1))
    # dims [xyz, ch]

    log_module.debug(f"start processing")
    kernel_size = 20

    # calculate HiP
    # for aspire we need two echo images obeying m * TE_j = (m + 1) TE_k, for m int
    # which would be true for all spin echoes. in case of gre and bipolar readouts for megesse / grase,
    # we have an additional offset term for the central echo TE_se + m * delta_TE_j = TE_se + (m + 1) * delta_TE_k
    # we might be able to include
    # for now we pick first three spin echoes

    # caution! this needs to be adjsuted for each sequence
    se_indices = [*se_indices]
    m_jk = np.moveaxis(m_j, -1, 0)[se_indices]
    phi_jk = np.moveaxis(phi_j, -1, 0)[se_indices]

    m_int = 1  # aspire condition

    log_module.debug("calculate HiP")
    # combine multiple echoes 1-2 & 2-3 & 3 - 4
    # delta_phi_kj = np.zeros((num_combinations, *data.shape[:3]))
    # se 1-2, se 2-3, se 3 - 4
    # we reduce time dimension here! dims calculation [xyz - ch], result [3, xyz]
    delta_phi_kj = np.angle(
        np.sum(
            m_jk[0] * m_jk[1] * np.exp(
                1j * (phi_jk[1] - phi_jk[0])
            ),
            axis=-1
        )
    )

    log_module.debug(f"calculate phase offsets 0 - c")
    # calculate wrapped phase offsets, subtract from j phase, j is first echo (1, 2) k is second used echo (2, 3)
    # dims phi_jk [xyz, ch], delta_phi_jk [xyz]
    phi_0_c = helper_fns.apply_phase_wrap(
        phi_jk[0] - m_int * delta_phi_kj[:, :, :, None]
    )

    # smooth wrapped phase offsets
    log_module.debug(f"filter calculated offsets phi_0_c")
    phi_0_c_filtered = sndi.gaussian_filter(phi_0_c, sigma=kernel_size, axes=(0, 1))

    log_module.info(f"\t\tset data for phase combine")
    # phase combine coils - j index of echo number
    # dims [xyz - ch]
    m_j = (m_weights ** 2)[:, :, :, :, None]  # square for weighting
    phi_diff = phi_j - phi_0_c_filtered[:, :, :, :, None]
    # dims phi_0_filtered [xyz - ch]
    log_module.info(f"\t\tphase combine coils")
    # want result dims [xyz, t]
    data_phase_combined = np.angle(
        np.sum(
            m_j * np.exp(1j * phi_diff),
            axis=-2
        )
    )

    log_module.debug(f"calculate gradient phase offset")
    # build some differences to compute the rest. only useful for megesse, switched readout directions
    # (phi_3 - phi_2) - (phi_2 - phi_1)
    diff_phi_kj = data_phase_combined[:, :, :, 2:4] - data_phase_combined[:, :, :, 1:3]
    grad_phase_offset = - (diff_phi_kj[:, :, :, 0] - diff_phi_kj[:, :, :, 1]) / 4

    return data_phase_combined, grad_phase_offset
