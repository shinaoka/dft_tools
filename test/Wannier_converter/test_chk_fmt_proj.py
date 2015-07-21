from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive import HDFArchive

from os.path import isfile
import subprocess
import numpy
import pytriqs.utility.mpi as mpi

input_file="SrVO3" # checkpoint is stored in SrVO3.chk.fmt

# remove SrVO3.h5 if it exists
if mpi.is_master_node():
     if isfile(input_file+".h5"):
        subprocess.call(["rm",input_file+".h5"])
mpi.barrier()

# initialize converter
converter=Wannier90Converter(filename=input_file)

# read SrVO3.win, SrVO3.chk.fmt, SrVO3_hr.dat to get  projections
converter.convert_dft_input()

# load projections from the reference file
hdf_proj = HDFArchive('SrVO3_proj.h5','r')

success=True


# check projections
if not  hdf_proj["SumK_DFT"]["proj_mat"].shape==converter.proj_mat.shape and numpy.allclose(hdf_proj["SumK_DFT"]["proj_mat"], converter.proj_mat):
    converter.make_statement("Test Failed!  hdf_proj['proj_mat'] is different than converter.proj_mat!")
    success=False

del hdf_proj

if success:
    converter.make_statement("Constructing projections: test successful!")
else:
    converter.make_statement("Constructing projections: test failed!")

# check shells
valid_shells=[{'sort': 0, 'dim': 3, 'l': 2, 'atom': 0}]
if len(converter.shells)==1 and cmp(valid_shells[0],converter.shells[0])==0:
    converter.make_statement("Constructing shells: test successful!")
else:
    converter.make_statement("Constructing shells: test failed!")

# check correlated shells
valid_corr_shells=[{'sort': 0, 'dim': 3, 'l': 2, 'irep': 0, 'SO': 0, 'atom': 0}]
if len(converter.corr_shells)==1 and cmp(valid_corr_shells[0],converter.corr_shells[0])==0:
    converter.make_statement("Constructing correlated shells: test successful!")
else:
    converter.make_statement("Constructing correlated shells: test failed!")