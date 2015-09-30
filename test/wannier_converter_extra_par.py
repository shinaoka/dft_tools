from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive.hdf_archive import HDFArchive

#    Test for reading extra parameters by Wannier90Converter
#
#    Input files: wannier_converter_extra.par, wannier_converter_extra_par.py
#
#    Output files: wannier_converter_extra_par.output.h5
#
#    Command: pytriqs wannier_converter_extra_par.py
#
#    Description:
#    A simple benchmark for reading additional parameters by Wannier90converter.
#
#
#    Tested features within this benchmark:
#
#        * reading all supported keywords
#       
#        * various variants of comments

input_file = "wannier_converter"  # Core part of the name of file here. File input_file+"_extra.par" will be opened

converter = Wannier90Converter(filename=input_file)

ar = HDFArchive('wannier_converter_extra_par.output.h5', 'w')
ar["num_zero"] = converter._num_zero
ar["verbosity"] = converter._verbosity
ar["non_standard_corr_shells"] = converter._non_standard_corr_shells
del ar

