import sys
import pathlib as plib

# make twixtools source root available
cwd = plib.Path(__file__).absolute().parent.parent
twixtools_path = cwd.joinpath("twixtools")
pypsi_path = cwd.joinpath("pypulseq_interface")
sys.path.append(twixtools_path.as_posix())
sys.path.append(pypsi_path.as_posix())
