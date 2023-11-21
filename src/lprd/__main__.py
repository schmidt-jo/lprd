import logging

import nibabel as nib
import numpy as np
import torch

from lprd import lprd_io, options, plotting, recon_fns


def save_k_space_torch(k_img: np.ndarray, k_mask: np.ndarray, affine: np.ndarray,
                                     opts: options.Config, name_add: str = ""):
    if name_add:
        if not name_add.endswith("_"):
            name_add = name_add + "_"
    # save k-space
    lprd_io.save_torch_tensor(
        tensor=torch.from_numpy(k_img.copy()), opts=opts, name=f"{name_add}k_space",
    )
    lprd_io.save_torch_tensor(
        tensor=torch.from_numpy(affine.copy()), opts=opts, name=f"{name_add}affine",
    )
    lprd_io.save_torch_tensor(
        tensor=torch.from_numpy(k_mask.copy()), opts=opts, name=f"{name_add}sampling_mask",
    )


def save_and_recon_k_space_nii(k_img: np.ndarray, k_mask: np.ndarray, affine: np.ndarray,
                               opts: options.Config, name_add: str = "",
                               header: nib.Nifti1Header = nib.Nifti1Header(),
                               combine_phase: bool = True):
    if name_add:
        if not name_add.endswith("_"):
            name_add = name_add + "_"

    # so far for testing, naive recon and save
    if opts.seq_type == "mese_fidnav":
        se_indices = (0, 1)
    else:
        se_indices = (2, 5)
    # k img dims [x,y,z,ch,echoes]
    naive_recon_mag, naive_recon_phase, grad_offset = recon_fns.fft_mag_sos_phase_aspire_recon(
        k_data_xyz_ch_t=k_img, se_indices=se_indices, combine_phase=combine_phase
    )
    lprd_io.save_nii(
        data=naive_recon_mag, affine=affine, opts=opts, name=f"{name_add}img_naive_recon_mag", header=header,
        max_scale_to=1000.0
    )
    lprd_io.save_nii(
        data=naive_recon_phase, affine=affine, opts=opts, name=f"{name_add}img_naive_recon_phase", header=header
    )
    lprd_io.save_nii(
        data=k_mask, affine=affine, opts=opts, name=f"{name_add}k_space_sampling_mask", max_scale_to=1, header=header
    )
    # save grad offset for bipolar sequence
    if not opts.seq_type == "mese_fidnav":
        lprd_io.save_nii(
            data=grad_offset, affine=affine, opts=opts, name=f"{name_add}bipolar_grad_offset", header=header
        )


def main(opts: options.Config):
    # check files provided, check output path exist ifn make
    opts.check_paths()
    k_space, k_mask, k_nav, k_nav_mask, aff, aff_nav, interface = lprd_io.load_raw_data_and_interface(
        opts=opts
    )
    k_nii_hdr = lprd_io.build_nii_header_info(interface=interface, affine=aff)

    if opts.visualize:
        plotting.visualize_k_space(k_data_xyz_ch_t=k_mask[:, :, None, None], name="sampling_mask",
                                   opts=opts, log_mag=False)
        plotting.visualize_k_space(k_data_xyz_ch_t=k_space, opts=opts, name="k_space_data")
        plotting.visualize_naive_recon(k_data_xyz_ch_t=k_space, opts=opts, name="naive_recon")
        if k_nav is not None:
            plotting.visualize_k_space(k_data_xyz_ch_t=k_nav[:, :, :, :, None], opts=opts, name="nav_data")
            plotting.visualize_k_space(k_data_xyz_ch_t=k_nav_mask[:, :, None, None, None], name="nav_sampling_mask",
                                       opts=opts, log_mag=False)
            plotting.visualize_naive_recon(k_data_xyz_ch_t=k_nav[:, :, :, :, None], opts=opts, name="naive_recon_nav")

    # save k-space
    save_and_recon_k_space_nii(k_img=k_space, opts=opts, affine=aff, k_mask=k_mask, header=k_nii_hdr)
    save_k_space_torch(k_img=k_space, opts=opts, affine=aff, k_mask=k_mask)

    if k_nav is not None:
        k_nav_nii_hdr = lprd_io.build_nii_header_info(interface=interface, affine=aff_nav, nav=True)
        save_k_space_torch(k_img=k_nav, opts=opts, affine=aff_nav, k_mask=k_mask, name_add="nav")
        save_and_recon_k_space_nii(
            k_img=k_nav, opts=opts, affine=aff_nav, k_mask=k_mask, name_add="nav", header=k_nav_nii_hdr,
            combine_phase=False
        )


if __name__ == '__main__':
    # create cli parser
    parser, prog_args = options.create_cli()
    if prog_args.config.debug_flag:
        level = logging.DEBUG
    else:
        level = logging.INFO
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=level)
    logging.info("_________________________________________________________")
    logging.info("____________ python - twix - raw data loader ____________")
    logging.info("_________________________________________________________")
    prog_opts = options.Config.from_cli_args(prog_args)
    prog_opts.display_settings()

    # run script
    try:
        main(prog_opts)
    except Exception as e:
        logging.exception(e)
        parser.print_usage()
        exit(-1)
    logging.info(f"Processing finished successful!")
