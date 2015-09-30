from pytriqs.archive import HDFArchive
from pytriqs.applications.dft.sumk_dft import SumkDFT
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from os.path import isfile
import subprocess


# Test for the initial chemical potential determination
#
# Input files: wannier_converter_SrVO3.win, wannier_converter_SrVO3_hr.dat, wannier_converter_SrVO3.chk.fmt,
#              wannier_converter_dft_dmft_cthyb_chem_pot.py
#
# Output files: wannier_converter_dft_dmft_cthyb_chem_pot.output.h5
#
# Command: pytriqs wannier_converter_dft_dmft_cthyb_chem_pot.py
#
# Description:
# Wannier90Converter prepares input for SumkDFT. Chemical
# potential for the initial DMFT iteration is calculated by SumkDFT from data submitted by Wannier90Converter.
#
# Tested features  within this benchmark:
#
#     * It is verified if initial chemical potential for the self energy equal zero is valid.
#

dft_filename = 'wannier_converter_SrVO3'
use_blocks = True                # use bloc structure from DFT input
h_field = 0.0
chemical_potential = 12.5  # to speed up a benchmark
prec_mu = 0.0001

# remove SrVO3.h5 if it exists, in the benchmark we test scenario: calculation from scratch
if isfile(dft_filename + ".h5"):
    subprocess.call(["rm", dft_filename + ".h5"])

Converter = Wannier90Converter(filename=dft_filename, repacking=True)
Converter.convert_dft_input()
SK = SumkDFT(hdf_file=dft_filename + '.h5', use_dft_blocks=use_blocks, h_field=h_field, dft_data="SumK_DFT")
SK.set_mu(chemical_potential)
SK.calc_mu(prec_mu)

ar = HDFArchive("wannier_converter_dft_dmft_cthyb_chem_pot.output.h5", "w")
ar["chemical_potential"] = SK.chemical_potential
del ar
