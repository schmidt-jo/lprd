import torchkbnufft as tkbn
from lprd import options, plotting, helper_fns
import numpy as np
import torch
import logging
import pathlib as plib
from pypulseq_interface import pypsi
import pandas as pd
import twixtools as tt
import nibabel as nib
import typing

log_module = logging.getLogger(__name__)


def check_f_exists(f_name: plib.Path):
    if not f_name.is_file():
        err = f"Provided {f_name} does not point to a file"
        log_module.exception(err)
        raise FileNotFoundError(err)


def load_twix(opts: options.Config) -> (tt.geometry, list):
    # load in data
    file_path = plib.Path(opts.raw_data_file).absolute()
    check_f_exists(file_path)
    log_module.info(f"loading twix file: {file_path}")
    twix = tt.read_twix(file_path.as_posix(), parse_geometry=True, parse_data=True)[-1]
    geom = twix["geometry"]
    mdb_list = twix["mdb"]
    return geom, mdb_list


def load_interface_file(opts: options.Config) -> pypsi.Params:
    log_module.debug("load in interface file")
    interface = pypsi.Params.load(opts.pyp_interface_file)
    interface.config.output_path = opts.output_path
    return interface


def load_raw_data_and_interface(opts: options.Config, scale_to_max_val: float = 1000.0):
    """ returns k-space-image-data, k-space-sampling-mask,
        k-space-navigator-data (if applicable), k-space-navigator-mask (if applicable)
        affine-matrix, affine-nav-matrix (if-applicable)
    """
    log_module.debug("load in data")
    geom, mdb_list = load_twix(opts=opts)
    interface = load_interface_file(opts=opts)

    # create shorthand
    img_params = interface.recon.multi_echo_img
    nav_params = interface.recon.navigator_img

    if opts.visualize:
        interface.visualize()

    log_module.debug(f"setup recon info")
    num_coils = mdb_list[-1].mdh.UsedChannels

    log_module.debug(f"allocate img arrays")
    etl = np.max(interface.sampling_k_traj.sampling_pattern["echo_num"].unique()) + 1
    if etl > img_params.etl:
        warn = f"recon interface set etl ({img_params.etl}) differs from num sampling pattern " \
               f"echo identifiers ({etl}), this might be okay if theres multiple sampling per readout pulse," \
               f" eg. megesse sequences"
        log_module.warning(warn)
    etl = np.max([etl, img_params.etl])

    # build image array
    # x is rl, y is pa
    if interface.pypulseq.acq_phase_dir == "RL":
        transpose_xy = True
    elif interface.pypulseq.acq_phase_dir == "PA":
        transpose_xy = False
    else:
        err = f"pypulseq phase encode direction {interface.pypulseq.acq_phase_dir} not recognized"
        log_module.exception(err)
        raise AttributeError(err)
    k_space = np.zeros(
        (img_params.n_read * img_params.os_factor, img_params.n_phase, img_params.n_slice, num_coils, etl),
        dtype=complex
    )
    k_sampling_mask = np.zeros(
        (img_params.n_read, img_params.n_phase, etl),
        dtype=bool
    )

    log_module.debug(f"allocate navigator arrays")
    nav = nav_params.n_read > 0
    if nav:
        nav_shape = (nav_params.n_read * nav_params.os_factor, nav_params.n_phase, nav_params.n_slice, num_coils)
        k_nav = np.zeros(nav_shape, dtype=complex)
        k_nav_mask = np.zeros(nav_shape[:2], dtype=bool)
    else:
        k_nav = None
        k_nav_mask = None

    # setup tkb
    # use tkbnufft interpolation to get k-space data from trajectory
    device = torch.device("cpu")
    # we want to regrid the data to the regular grid but have oversampling in read direction
    img_size = (img_params.os_factor * img_params.n_read,)
    grid_size = (img_params.os_factor * img_params.n_read,)
    tkbn_interp = tkbn.KbInterpAdjoint(
        im_size=img_size,
        grid_size=grid_size,
        device=device
    )

    log_module.debug("loop through acquisition types")
    for acq_type in interface.sampling_k_traj.k_trajectories["acquisition"].unique():
        log_module.info(f"processing {acq_type} acquisition data")
        # get all sampling pattern entries matching acq. type and no navigator
        sp_sub = interface.sampling_k_traj.sampling_pattern[
            (interface.sampling_k_traj.sampling_pattern["acq_type"] == acq_type)
        ]
        if nav:
            # split img data and navigators
            sp_img = sp_sub[~sp_sub["nav_acq"]]
            sp_nav = sp_sub[sp_sub["nav_acq"]]
        else:
            sp_img = sp_sub
            sp_nav = pd.DataFrame()

        # need trajectory
        k_traj_read_line = torch.from_numpy(
            interface.sampling_k_traj.k_trajectories[
                interface.sampling_k_traj.k_trajectories["acquisition"] == acq_type
                ]["k_traj_position"].to_numpy()
        )
        # get line numbers
        line_nums = sp_img.index.to_list()
        if line_nums:
            log_module.debug(f"\t\t___process img data")
            # get indices of entries
            ids_echo = torch.from_numpy(sp_img["echo_num"].to_numpy()).to(torch.int)
            ids_slice = torch.from_numpy(sp_img["slice_num"].to_numpy()).to(torch.int)
            ids_pe = torch.from_numpy(sp_img["pe_num"].to_numpy()).to(torch.int)

            log_module.info(f"\t\t - fill sampling mask")
            ids_pe_slice = torch.from_numpy(sp_img[sp_img["slice_num"] == 0]["pe_num"].to_numpy()).to(torch.int)
            ids_echo_slice = torch.from_numpy(sp_img[sp_img["slice_num"] == 0]["echo_num"].to_numpy()).to(torch.int)
            # k_sampling_mask[:, ids_pe_slice, ids_echo_slice] = 1

            log_module.info(f"\t\t - fill k-space data")
            data = torch.from_numpy(np.array([mdb_list[i].data.T for i in line_nums]))
            # want it in dimensions [N (batch), C (coils), k-length]
            data = torch.moveaxis(data, -1, 1)
            # small sanity check
            if data.shape[0] != len(sp_sub):
                err = "picked raw data size does not match number of sampling pattern entries"
                log_module.exception(err)
                raise AttributeError(err)

            if opts.grid_data:
                # from -pi to pi -- radians / voxel
                k_traj_read_line *= 2 * torch.pi
                traj_len = k_traj_read_line.shape[0]
                # calculate density compensation
                # log_module.info("\t\t - set density compensation")
                # dcomp = tkbn.calc_density_compensation_function(k_traj_read_line[None, :], im_size=img_size,
                #                                                 num_iterations=10)
                dcomp = 1.0

                # fill entries of trajectory    -> need dim [N (batch{, len(grid_size) == 1, k-length]
                k_traj_arr = torch.zeros((len(sp_sub), 1, traj_len))
                k_traj_arr[:, 0, :] = k_traj_read_line[None, :]
                # reshape to batch size
                if opts.visualize:
                    # peek at data
                    k_plot = torch.zeros(
                        (traj_len, img_params.n_phase, img_params.n_slice, num_coils, etl),
                        dtype=data.dtype
                    )
                    k_plot[:, ids_pe, ids_slice, :, ids_echo] = torch.moveaxis(data, -1, 1)
                    plotting.visualize_k_space(k_data_xyz_ch_t=k_plot.numpy(), opts=opts,
                                               name=f"test_data_{acq_type}_before_gridding")

                # set device
                k_traj_arr = k_traj_arr.to(device)
                data = data.to(device)

                # interpolate data
                log_module.info("\t\t - tkbn interpolation")
                data_interp = torch.fft.fftshift(tkbn_interp(dcomp * data, k_traj_arr).conj(), dim=-1)
                # we now have a k-space which should be having read direction interpolated from the trajectories.

                log_module.info(f"\t\t - assign to k-space")
                k_space[:, ids_pe, ids_slice, :, ids_echo] = torch.moveaxis(
                    data_interp.resolve_conj(), -1, 1
                ).numpy(force=True)

                # drop edges in pf region not acquired -> produces edge artifacts in interpolation
                if "pf" in acq_type:
                    # sanity check if 0 filled pf region starts at this side of the readout
                    # look at center of phase. supposed to be fully sampled
                    y = int(img_params.n_phase / 2)
                    # we extract nonzero elements of all respective entries (ids slice and echo)
                    # along the read direction and central phase encode
                    ids_nonzero_read = np.nonzero(np.abs(k_space[:, y, ids_slice, :, ids_echo]))[1]
                    # get unique entries
                    ids_nonzero_read_unique = np.unique(ids_nonzero_read)
                    # if pf not acquired region at lower side of k - space, the first two entries should have a bigger gap
                    # want to update sampling mask for pf readout, also extract 0 sampling reads
                    if np.diff(ids_nonzero_read_unique[:2]) > 3:
                        # 0 filled pf starts at x = 0, set outer line and first sampled line 0 to prevent edge artifacts
                        k_space[ids_nonzero_read_unique[0], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_space[ids_nonzero_read_unique[1], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_sampling_mask[:ids_nonzero_read_unique[1] + 1, ids_pe, ids_echo] = 0
                    elif np.diff(ids_nonzero_read_unique[-2:]) > 3:
                        # 0 filled part starts at x = img_read_n and stretches from this side towards central k space,
                        # set outer line and first sampled line 0 to prevent edge artifacts
                        k_space[ids_nonzero_read_unique[-1], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_space[ids_nonzero_read_unique[-2], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_sampling_mask[ids_nonzero_read_unique[-2:], ids_pe, ids_echo] = 0
                    else:
                        err = f"set partial fourier acquisition but couldn't find 0-filled outer k-space"
                        log_module.exception(err)
                        AttributeError(err)
            else:
                # no gridding
                # calculate the closest points on grid from trajectory. get positions
                k_traj_pos = (0.5 + k_traj_read_line) * img_params.n_read * img_params.os_factor
                # just round to the nearest point in 1d
                k_traj_pos = k_traj_pos.round().to(torch.int)
                # find indices of points that are actually within our k-space in case the trajectory maps too far out
                k_traj_valid_index = torch.squeeze(
                    torch.nonzero((k_traj_pos >= 0) &
                                  (k_traj_pos < img_params.os_factor * img_params.n_read)))
                # find positions of those indices
                k_traj_valid_pos = k_traj_pos[k_traj_valid_index].to(torch.int)
                indices_in_k_sampling_img = (
                        k_traj_valid_pos[::img_params.os_factor] / img_params.os_factor
                ).to(torch.int).numpy()
                # pick valid data
                k_no_grid_sub = data[:, :, k_traj_valid_index]
                # fill at respective positions
                k_space[
                    k_traj_valid_pos[None, :],
                    ids_pe[:, None], ids_slice[:, None],
                    :, ids_echo[:, None]
                ] = torch.moveaxis(k_no_grid_sub, -1, 1).numpy(force=True)
                k_sampling_mask[indices_in_k_sampling_img[:, None], ids_pe_slice[None, :], ids_echo_slice[None, :]] = 1

        # process navigators - no gridding
        # get line numbers
        line_nums = sp_nav.index.to_list()
        if line_nums:
            log_module.debug(f"\t\t___process nav data")
            # get indices of entries
            ids_slice = torch.from_numpy(sp_nav["slice_num"].to_numpy()).to(torch.int)
            ids_pe = torch.from_numpy(sp_nav["pe_num"].to_numpy()).to(torch.int)

            log_module.info(f"\t\t - fill sampling mask")
            ids_pe_slice = torch.from_numpy(sp_nav[sp_nav["slice_num"] == 0]["pe_num"].to_numpy()).to(torch.int)
            k_nav_mask[:, ids_pe_slice] = 1

            log_module.info(f"\t\t - fill k-space data")
            data = torch.from_numpy(np.array([mdb_list[i].data.T for i in line_nums]))
            # want it in dimensions [N (batch), C (coils), k-length]
            data = torch.moveaxis(data, -1, 1)
            # small sanity check
            if data.shape[0] != len(sp_sub):
                err = "picked raw data size does not match number of sampling pattern entries"
                log_module.exception(err)
                raise AttributeError(err)
            # no gridding
            # calculate the closest points on grid from trajectory. get positions
            k_traj_pos = (0.5 + k_traj_read_line) * nav_params.n_read * nav_params.os_factor
            # just round to the nearest point in 1d
            k_traj_pos = k_traj_pos.round().to(torch.int)
            # find indices of points that are actually within our k-space
            k_traj_pos_valid = torch.squeeze(
                torch.nonzero((k_traj_pos >= 0) &
                              (k_traj_pos < nav_params.os_factor * nav_params.n_read)))
            # find positions of those indices, skip duplicates
            k_traj_pos_valid = k_traj_pos[k_traj_pos_valid]
            # pick valid locations of data and remove os
            # k_traj_pos_valid = k_traj_pos_valid_rm_os[k_traj_pos_valid_rm_os % nav_params.os_factor == 0]
            k_no_grid_sub = data[:, :, k_traj_pos_valid]
            # fill at respective positions
            k_nav[k_traj_pos_valid[None, :], ids_pe[:, None], ids_slice[:, None], :] = torch.moveaxis(
                k_no_grid_sub, -1, 1
            ).numpy(force=True)

    log_module.info(f"\t -- done!")

    # remove oversampling
    k_space = helper_fns.remove_os(
        data=k_space, data_input_sampled_in_time=True, read_dir=0, os_factor=img_params.os_factor
    )
    # fft bandpass filter not consistent with undersampled data, eg. partial fourier, remove artifacts
    k_space *= k_sampling_mask[:, :, None, None, :]

    # correct gradient directions
    # k_space = np.flip(k_space, axis=(0, 1, 2))
    # k_sampling_mask = np.flip(k_sampling_mask, axis=(0, 1))

    # scale values
    ks_max = np.max(np.abs(k_space))
    k_space *= scale_to_max_val / ks_max

    # get affine
    pix_scaling_mat = np.diag([
            interface.recon.multi_echo_img.resolution_read,
            interface.recon.multi_echo_img.resolution_phase,
            interface.recon.multi_echo_img.resolution_slice
        ])

    if nav:
        # remove oversampling
        k_nav = helper_fns.remove_os(
            data=k_nav, data_input_sampled_in_time=True, read_dir=0, os_factor=nav_params.os_factor
        )
        # correct gradient directions
        # k_nav = np.flip(k_nav, axis=(0, 1, 2))
        # k_nav_mask = np.flip(k_nav_mask, axis=(0, 1))

        # scale values
        kn_max = np.max(np.abs(k_nav))
        k_nav *= scale_to_max_val / kn_max
    # swap dims if phase dir RL
    if transpose_xy:
        k_space = np.swapaxes(k_space, 0, 1)
        k_sampling_mask = np.swapaxes(k_sampling_mask, 0, 1)
        pix_scaling_mat = np.swapaxes(pix_scaling_mat, 0, 1)
        if nav:
            k_nav = np.swapaxes(k_nav, 0, 1)
            k_nav_mask = np.swapaxes(k_nav_mask, 0, 1)

    aff = get_affine(
        geom,
        scaling_mat=pix_scaling_mat
    )
    if nav:
        # navigators scaled (lower resolution)
        nav_scale_x = k_space.shape[0] / k_nav.shape[0]
        nav_scale_y = k_space.shape[1] / k_nav.shape[1]
        scale_mat = np.matmul(
            np.diag([nav_scale_x, nav_scale_y, 1]),
            pix_scaling_mat
        )
        aff_nav = get_affine(geom, scaling_mat=scale_mat)
    else:
        aff_nav = None

    return k_space, k_sampling_mask, k_nav, k_nav_mask, aff, aff_nav, interface


def save_echo_types(et: pd.DataFrame, opts: options.Config, name_add: str = ""):
    if name_add:
        if not name_add.endswith("_"):
            name_add = name_add + "_"
    # build path
    out_path = plib.Path(opts.output_path).absolute()
    if out_path.suffixes:
        out_path = out_path.parent
    out_path.mkdir(parents=True, exist_ok=True)
    et_path = out_path.joinpath(f"{name_add}echo_types").with_suffix(".json")
    log_module.info(f"writing file: {et_path}")
    et.to_json(et_path.as_posix(), indent=2)


def save_torch_tensor(tensor: torch.tensor, opts: options.Config, name: str):
    file_name = plib.Path(opts.output_path).absolute().joinpath(name).with_suffix(".pt")
    log_module.info(f"writing file: {file_name}")
    torch.save(tensor, file_name.as_posix())
    log_module.debug(f"verifying save")
    test = torch.load(file_name)
    assert torch.allclose(test, tensor)


def get_affine(geom: tt.geometry.Geometry, scaling_mat: np.ndarray = np.eye(3)):
    aff_rot = np.array(geom.rotmatrix)
    aff_comb = np.matmul(aff_rot, scaling_mat)
    aff_translation = np.array(geom.offset)

    aff_matrix = np.zeros((4, 4))
    aff_matrix[:3, :3] = aff_comb
    aff_matrix[:-1, -1] = aff_translation
    aff_matrix[-1, -1] = 1.0
    return aff_matrix


def build_nii_header_info(interface: pypsi.Params, affine: np.ndarray, nav: bool = False) -> nib.Nifti1Header:
    # create header template
    header = nib.Nifti1Header()
    # choose interface recon info
    recon_info = interface.recon.multi_echo_img
    if nav:
        recon_info = interface.recon.navigator_img
    # set affine
    header.set_qform(affine=affine)
    header.__setitem__("qform_code", 1)
    header.set_sform(affine=affine)
    header.__setitem__("sform_code", 1)
    # set pixdims
    header.set_xyzt_units("mm", "sec")
    return header


def save_nii(opts: options.Config, name: str, data: typing.Union[torch.tensor, np.ndarray],
             affine: typing.Union[torch.tensor, np.ndarray], max_scale_to: float = None,
             header: nib.Nifti1Header = nib.Nifti1Header()):
    # strip file ending
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".nii"):
        name = name[:-4]
    # swap dots in name
    name = name.replace(".", "p")
    # build path
    out_path = plib.Path(opts.output_path).absolute()
    nii_out = out_path.joinpath(name).with_suffix(".nii")
    if out_path.suffixes:
        out_path = out_path.parent
    out_path.mkdir(parents=True, exist_ok=True)
    # check datatype
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(affine):
        affine = affine.numpy()
    if max_scale_to is not None:
        data = np.divide(data, np.max(np.abs(data))) * max_scale_to
    # if data complex split into mag and phase
    if np.iscomplexobj(data):
        mag = np.abs(data)
        phase = np.angle(data)
        save_dat = [mag, phase]
        save_labels = ["_mag", "_phase"]
    else:
        save_dat = [data]
        save_labels = [""]
    # save
    for dat_idx in range(save_dat.__len__()):
        f_name = nii_out.with_stem(f"{nii_out.stem}{save_labels[dat_idx]}")
        log_module.info(f"Writing file: {f_name.as_posix()}")
        img = nib.Nifti1Image(save_dat[dat_idx], affine=affine, header=header)
        nib.save(img, f_name.as_posix())
