from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive import HDFArchive

from os.path import isfile
import subprocess
import numpy
import pytriqs.utility.mpi as mpi


# Test for validity of reading shells and projections  from *.chk.fmt file
#
# Input files: wannier_converter_SrVO3.win, wannier_converter_SrVO3_hr.dat, wannier_converter_SrVO3.chk.fmt wannier_converter_test_chk_fmt_shells.py
#
# Outputfiles: wannier_converter_test_chk_fmt_shells.output.h5
#
# Command: pytriqs wannier_converter_test_chk_fmt_shells.py
#
# Descriprion:
# Converter is initialized and information from wannier_converter_SrVO3.win,
# wannier_converter_SrVO3_hr.dat, wannier_converter_SrVO3.chk.fmt is read.

# wannier_converter_SrVO3_hr.dat is one of the output files of wannier90. wannier_converter_SrVO3_hr.dat stores a real representation of the  local
# Hamiltonian expressed in MLWFs for SrVO3 with t2g states taken into account. Basing on read data, shells,
# correlated shells and projections (U matrices)  are constructed for SrVO3.
#
# Tested features  within this benchmark:
#
#     * It is verified if  shells are constructed properly,
#
#     * It is verified if correlated shells are constructed properly.
#
#     * It is verified if projections are constructed properly


input_file="wannier_converter_SrVO3" # checkpoint is stored in wannier_converter_SrVO3.chk.fmt

# remove wannier_converter_SrVO3.h5 if it exists
if isfile(input_file+".h5"):
    subprocess.call(["rm",input_file+".h5"])

# initializes converter
converter=Wannier90Converter(filename=input_file)

# reads wannier_converter_SrVO3.win, evaluates shells without sort
converter._read_win_file()

# reads wannier_converter_SrVO3_hr.dat, adds sort to shells, evaluates correalted shells
converter._h_to_triqs()

# reads  wannier_converter_SrVO3.chk.fmt and constructs projections
converter._produce_projections()

ar= HDFArchive('wannier_converter_test_chk_fmt_shells.output.h5','w')
ar["shells"]=converter.shells
ar["corr_shells"]=converter.corr_shells
ar["proj_mat"]=converter.proj_mat
del ar
