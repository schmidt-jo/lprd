import sys
import pathlib as plib

lprd_path = plib.Path(__file__).absolute().parent.parent
twix_path = lprd_path.joinpath("twixtools")
pygrappa_path = lprd_path.joinpath("pygrappa")
sys.path.append(lprd_path.as_posix())
sys.path.append(twix_path.as_posix())
sys.path.append(pygrappa_path.as_posix())
import os

os.environ["OMP_NUM_THREADS"] = "160"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "160"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "160"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "160"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "160"  # export NUMEXPR_NUM_THREADS=6
import logging
import numpy as np
import tqdm
from twixtools import twixtools as tt
import pandas as pd
import plotly.express as px
from lprd import helper_fns, lprd_io
import pygrappa
import nibabel as nib


def main():
    path = plib.Path(
        "/data/pt_np-jschmidt/data/01_in_vivo_scan_data/paper_protocol_mese/7T/"
        "2023-10-30/raw/meas_MID00044_FID27156_mod_semc_r0p7_fa140_z4000_pat3_36_sl35_200_esp9p5.dat"
    ).absolute()
    name = "mod_semc_r0p7_fa140_z4000_pat3_36_sl35_200_esp9p5"
    out_path = path.parent

    logging.info("__ Loading")
    logging.info(f"\tload file: {path.as_posix()}")
    twix = tt.read_twix(path.as_posix())[0]
    logging.info(f"\tassigning data")

    geom = twix["geometry"]
    mdbs = twix["mdb"]
    hdr = twix["hdr"]
    meas_dict = {}

    logging.info("\t\t-extract Meas info")
    for key, val in hdr["Meas"].items():
        if val:
            meas_dict.__setitem__(key, val)
    pd_meas = pd.Series(meas_dict)
    pd_meas_yaps_slice = pd.Series(hdr["MeasYaps"]["sSliceArray"])

    n_echoes = pd_meas.lContrasts
    n_fe = int(pd_meas.iNoOfFourierColumns)
    n_fe_base = pd_meas.lBaseResolution
    n_pe = pd_meas.lPhaseEncodingLines
    n_slices = pd_meas_yaps_slice.lSize
    n_coils = int(pd_meas.iMaxNoOfRxChannels)

    os_factor = int(n_fe / n_fe_base)
    fa = pd_meas.FlipAngle

    tr = pd_meas.TR
    tes = pd_meas.alTE[:n_echoes]
    ref_lines_pe = pd_meas.lRefLinesPE
    acc_factor = pd_meas.lAccelFactPE

    fov_read = pd_meas.ReadFoV
    fov_phase = pd_meas.PhaseFoV
    slice_thickness = pd_meas_yaps_slice.asSlice[0]["dThickness"]
    slice_pos = []
    raw_data_flags = []
    noise_data_nums = []
    data_counter = 0

    for mdb in mdbs:
        z_pos = mdb.mdh.SliceData.SlicePos.Tra
        for flag in mdb.get_active_flags():
            if flag not in raw_data_flags:
                raw_data_flags.append(flag)
            if flag == "NOISEADJSCAN":
                noise_data_nums.append(data_counter)
                data_counter += 1
        if slice_pos:
            if np.min(np.abs(np.array(slice_pos) - z_pos)) > 1e-6:
                slice_pos.append(z_pos)
        else:
            slice_pos.append(z_pos)

    slice_pos_ordered = np.sort(slice_pos)
    slice_index_mapping = np.searchsorted(slice_pos_ordered, slice_pos)
    vox_size = np.array([fov_read, fov_phase, slice_thickness]) / np.array([n_fe_base, n_pe, 1.0])
    clins = 1 + np.max([mdb.cLin for mdb in mdbs])
    n_part = 1 + np.max([mdb.cPar for mdb in mdbs])

    logging.info(f"extract geometry / affine")
    # we need to get the gap to scale additionally
    slice_distance = slice_pos_ordered[1] - slice_pos_ordered[0]
    gap = slice_distance - vox_size[-1]
    scale_z = slice_distance / vox_size[-1]
    affine = lprd_io.get_affine(geom, scaling_mat=np.eye(3)*[1, 1, scale_z])

    n_ref_min = n_pe
    n_ref_max = 0

    k_space_img = np.zeros((n_fe, n_pe, n_slices, n_coils, n_echoes), dtype=complex)
    k_space_ref = np.zeros((n_fe, n_pe, n_slices, n_coils, n_echoes), dtype=complex)
    noise_data = []
    for k in tqdm.trange(len(mdbs), desc="process mdbs"):
        mdb = mdbs[k]
        flags = mdb.get_active_flags()
        if "SWAPPED" in flags:
            data = np.flip(mdb.data.T, 0)
        else:
            data = mdb.data.T
        pe = mdb.cLin
        t = mdb.cEco
        sli = mdb.cSlc

        if "NOISEADJSCAN" in flags:
            noise_data.append(data)
        elif mdb.is_image_scan():
            k_space_img[:, pe, slice_index_mapping[sli], :, t] = data
        else:
            k_space_ref[:, pe, slice_index_mapping[sli], :, t] = data
            if pe < n_ref_min:
                n_ref_min = pe
            if pe > n_ref_max:
                n_ref_max = pe
    logging.info("Pre-whiten / channel de-correlation")
    # pre whiten / noise decorrelation
    noise_data = np.squeeze(np.array(noise_data)).T
    # calculate noise correlation matrix, dims [nfe, nch].T
    noise_cov = np.cov(noise_data)
    test_cov = np.cov(k_space_img[:, 0, 0, :, 0].T)
    plot_data = np.concatenate((noise_cov[None], test_cov[None]), axis=0)
    fig = px.imshow(np.abs(plot_data), facet_col=0)
    fig_name = f"{name}_noise_cov"
    fig_file = out_path.joinpath(fig_name).with_suffix(".html")
    logging.info(f"\t\t - write channel covariance matrix plot: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())
    # calculate psi pre whitening matrix
    psi = np.matmul(noise_data, noise_data.T.conj()) / (noise_data.shape[1] - 1)
    psi_l = np.linalg.cholesky(psi)
    if not np.allclose(psi, np.dot(psi_l, psi_l.T.conj())):
        # verify that L * L.H = A
        err = "cholesky decomposition error"
        logging.error(err)
        raise AssertionError(err)
    psi_l_inv = np.linalg.inv(psi_l)
    # whiten data, psi_l_inv dim [nch, nch], k-space dims [nfe, npe, nsli, nch, nechos]
    k_space_img = np.einsum("ijkmn, lm -> ijkln", k_space_img, psi_l_inv, optimize=True)
    k_space_ref = np.einsum("ijkmn, lm -> ijkln", k_space_ref, psi_l_inv, optimize=True)

    logging.info("plot k-space slice")
    fig = px.imshow(
        np.swapaxes(
            np.log(
                np.abs(k_space_img[:, :, 10, 0]),
                where=np.abs(k_space_img[:, :, 10, 0]) > 1e-9,
                out=np.zeros_like(np.abs(k_space_img[:, :, 10, 0]))
            ),
            0, 1
        ),
        facet_col=-1, facet_col_wrap=4
    )
    fig_name = f"{name}_k-space-sort"
    fig_file = out_path.joinpath(fig_name).with_suffix(".html")
    logging.info(f"\t\t - write image k-space peek file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())
    fig = px.imshow(
        np.swapaxes(
            np.log(
                np.abs(k_space_ref[:, :, 10, 0]),
                where=np.abs(k_space_ref[:, :, 10, 0]) > 1e-9,
                out=np.zeros_like(np.abs(k_space_ref[:, :, 10, 0]))
            ),
            0, 1
        ),
        facet_col=-1, facet_col_wrap=4
    )
    fig_name = f"{name}_k-space-ref"
    fig_file = out_path.joinpath(fig_name).with_suffix(".html")
    logging.info(f"\t\t - write image k-space ref peek file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())

    # remove os - for development we want to save the naive recon.
    # hence we can save some fft'n with os removal
    lower_idx = int((os_factor - 1) / (2 * os_factor) * n_fe)
    upper_idx = int((os_factor + 1) / (2 * os_factor) * n_fe)
    # naive recon
    logging.info(f"save naive recon / remove os")
    logging.info(f"\t -- undersampled data")
    logging.info(f"\t\t - fft")
    # fft read dim
    img_data, _ = helper_fns.data_fft_to_time(data=k_space_img, axes=(0,))
    # remove os
    img_data = img_data[lower_idx:upper_idx]
    # get os removed k_space
    k_space_img, _ = helper_fns.data_fft_to_freq(data=img_data, axes=(0,))
    # for saving naive recon fft phase dim
    img_data, _ = helper_fns.data_fft_to_time(data=img_data, axes=(1,))
    # rsos
    logging.info(f"\t\t - rsos")
    img_naive_recon = np.sqrt(
        np.sum(
            np.square(
                np.abs(img_data)
            ),
            axis=-2
        )
    )
    nii_name = f"{name}_naive_recon_us_data"
    file_name = out_path.joinpath(nii_name).with_suffix(".nii")
    logging.info(f"\t\twrite file: {file_name.as_posix()}")
    img = nib.Nifti1Image(img_naive_recon, affine=affine)
    nib.save(img, file_name.as_posix())

    logging.info(f"\t -- reference data")
    logging.info(f"\t\t - fft")
    # fft read dir
    ref_data, _ = helper_fns.data_fft_to_time(data=k_space_ref, axes=(0,))
    # remove os
    img_ref = ref_data[lower_idx:upper_idx]
    # get os removed k_space
    k_space_ref, _ = helper_fns.data_fft_to_freq(data=img_ref, axes=(0,))
    # fft phase dir for saving naive recon
    img_ref, _ = helper_fns.data_fft_to_time(data=img_ref, axes=(1,))
    # rsos
    logging.info(f"\t\t - rsos")
    img_naive_recon = np.sqrt(
        np.sum(
            np.square(
                np.abs(img_ref)
            ),
            axis=-2
        )
    )
    nii_name = f"{name}_naive_recon_ref_data"
    file_name = out_path.joinpath(nii_name).with_suffix(".nii")
    logging.info(f"\t\twrite file: {file_name.as_posix()}")
    img = nib.Nifti1Image(img_naive_recon, affine=affine)
    nib.save(img, file_name.as_posix())

    # no dev we can use this:
    # k_space_img = helper_fns.remove_os(data=k_space_img, os_factor=os_factor)
    # k_space_ref = helper_fns.remove_os(data=k_space_ref, os_factor=os_factor)
    n_fe = k_space_img.shape[0]

    # grappa
    logging.info(f"start grappa processing")
    k_recon = np.zeros_like(k_space_img)
    # calibration data is the fully sampled region in the centre (here only dep. on phase encodes)
    input_acs = k_space_ref[:, n_ref_min:n_ref_max+1]
    acs_lines = input_acs.shape[1]

    for idx_slice in tqdm.trange(n_slices, desc='processing slices', leave=False, position=0):
        for idx_echo in tqdm.trange(n_echoes, desc='processing echos', leave=False, position=1):
            k_recon[:, :, idx_slice, :, idx_echo] = pygrappa.grappa(
                kspace=k_space_img[:, :, idx_slice, :, idx_echo], calib=input_acs[:, :, idx_slice, :, idx_echo],
                kernel_size=(6, 9), coil_axis=-1, lamda=0.01, memmap=False
            )
    # fft
    logging.info(f"fft to image")
    img_recon, _ = helper_fns.data_fft_to_time(k_recon, axes=(0, 1))
    logging.info(f"coil combine")
    rsos = np.sqrt(
        np.sum(
            np.square(
                np.abs(img_recon)
            ),
            axis=-2
        )
    )
    rsos *= 1000 / np.max(np.abs(rsos))
    nii_name = f"{name}_recon"
    file_name = out_path.joinpath(nii_name).with_suffix(".nii")
    logging.info(f"write file: {file_name.as_posix()}")
    img = nib.Nifti1Image(rsos, affine=affine)
    nib.save(img, file_name.as_posix())


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("_________________________________________________________")
    logging.info("____________ python - twix - raw data loader ____________")
    logging.info("_________________________________________________________")
    main()
