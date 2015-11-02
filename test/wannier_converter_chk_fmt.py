from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive.hdf_archive import HDFArchive

import numpy


# Test for validity of reading parameters from *.chk.fmt file
#
# Input files: wannier_converter_SrVO3.win, wannier_converter_SrVO3.chk.fmt, wannier_converter_chk_fmt.py
#
# Output files: wannier_converter_chk_fmt.output.h5
#
# Command: pytriqs wannier_converter_chk_fmt.py
#
# Description:
# In this benchmark a realistic case of SrVO3, with only V t2g orbitals included, is considered.
# It is verified if data is read correctly from formatted checkpoint file: wannier_converter_SrVO3.chk.fmt.
# wannier_converter_SrVO3.chk.fmt is a file which stores formatted (machine independent) data.
# wannier_converter_SrVO3.chk.fmt stores information required to restart the calculation of Wannier90 or
# enter the plotting phase which is performed by Wannier90 (from Wannier90  User Guide). In particular, it stores
# U matrices (called projections in TRIQS). A construction of projections is validated in the another benchmark.
#
# First data from wannier_converter_SrVO3.win is read (input file for Wannier90). Next data from
# wannier_converter_SrVO3.chk.fmt is read. wannier_converter_SrVO3.win is needed in this benchmark because while data
# from wannier_converter_SrVO3.chk.fmt is read, it is compared against already loaded data from
# wannier_converter_SrVO3.win. This is done in order to catch a potential incompatibility between win and
# checkpoint files. Please pay attention to name convention: if win file is called wannier_converter_SrVO3.win
# then formatted checkpoint file must be named wannier_converter_SrVO3.chk.fmt.
#
# Tested features within this benchmark:
#
#      * It is verified if the following keywords are read correctly:
#
#          - "checkpoint",
#
#          - "Have disentanglement",
#
#          - "Number of bands",
#
#          - "Number of exclude bands",
#
#          - "Number of Wannier orbitals",
#
#          - "M-P grid",
#
#          - "Number of kpoints",
#
#          - "Real lattice".
#
#

input_file = "wannier_converter_SrVO3"  # checkpoint is stored in wannier_converter_SrVO3.chk.fmt


converter = Wannier90Converter(filename=input_file)

# get data from *.win file
converter._read_win_file()

converter._initialize_U_matrices()

ar = HDFArchive('wannier_converter_chk_fmt.output.h5', 'w')
ar["chk_par"] = converter._read_chkpt_fmt(filename=input_file)  # get data from *.chk_fmt file
del ar

# correct_chk_file={'checkpoint': 'postwann',
#                   'Have disentanglement': False,
#                   'Number of bands': 3,
#                   'Number of exclude bands': 27,
#                   'Number of Wannier orbitals': 3,
#                   'mp_grid': [8, 8, 8],
#                   'Number of kpoints': 512,
#                   'Real lattice': numpy.array([[ 3.84209112,  0.        ,  0.        ],
#                                                [ 0.        ,  3.84209112,  0.        ],
#                                                [ 0.        ,  0.        ,  3.84209112]])}
#


