from pytriqs.archive import HDFArchive
from pytriqs.applications.dft.sumk_dft import SumkDFT
from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from os.path import isfile
import subprocess


# Test for validity of calculating the initial Weiss field (non-interacting Green's function for the impurity problem)
#
# Input files: wannier_converter_SrVO3.win, wannier_converter_SrVO3_hr.dat, wannier_converter_SrVO3.chk.fmt,
#              wannier_converter_dft_dmft_cthyb_extract_G_loc.py
#
# Output files: wannier_converter_dft_dmft_cthyb_extract_G_loc.output.h5
#
# Command: pytriqs wannier_converter_dft_dmft_cthyb_extract_G_loc.py
#
# Description:
# Wannier90Converter prepares input for SumkDFT. For the purpose of this benchmark the value of the chemical
# potential is assumed to be known before the start of the calculation. Within this benchmark initial value of Weiss
# field for the impurity solver is evaluated (in case of SrVO3 only one impurity problem has to be solved).
# .
# Tested features  within this benchmark:
#
#     * It is verified if initial Weiss field (non-interacting Green's function for the 1st
#       inequivalent correlated shell) is evaluated correctly.
#

dft_filename = 'wannier_converter_SrVO3'
use_blocks = True                # use bloc structure from DFT input
h_field = 0.0
chemical_potential = 12.5  # to speed up a benchmark search for mu is skipped and the final value is provided
beta = 40.0

# remove wannier_converter_SrVO3.h5 if it exists, in the benchmark we test scenario: calculation from scratch
if isfile(dft_filename + ".h5"):
    subprocess.call(["rm", dft_filename + ".h5"])

Converter = Wannier90Converter(filename=dft_filename)
Converter.convert_dft_input()

SK = SumkDFT(hdf_file=dft_filename + '.h5', use_dft_blocks=use_blocks, h_field=h_field, dft_data="SumK_DFT")
S = Solver(beta=beta, gf_struct=SK.gf_struct_solver[0])

SK.set_mu(chemical_potential)  # here we have skipped evaluation of the chemical potential and we give a final value
SK.put_Sigma(Sigma_imp=[S.Sigma_iw])
ar = HDFArchive("wannier_converter_dft_dmft_cthyb_extract_G_loc.output.h5", "w")
ar["G_loc"] = SK.extract_G_loc()[0]
del ar
