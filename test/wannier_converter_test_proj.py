from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive.hdf_archive import  HDFArchive


# Test for validity of constructed shells from *.win file
#
#    Input files: wannier_converter_proj.win, wannier_converter_test_proj.py,
#
#    Output files:  wannier_converter_test_proj.output.h5
#
#    Command: pytriqs wannier_converter_test_proj.py
#
#    Description:
#    In proj.win file hypothetical projections are stored. In this test it is verified whether
#    wannier90 converter can properly transform block:
#
#    begin projections
#    -----------------
#    -----------------
#
#    end projections
#
#    into shells (sort parameter is evaluated later basing on data from *_hr.dat file).
#
#    "------"  symbolizes valid entry with projection (or projections) written in the format accepted by wannier90.
#
#    Tested features within this benchmark:
#
#       * Various possible formats of projections are used (chapter 3 from Wannier90 User Guide).
#
#       * It is checked if comment sections are correctly detected.
#
#       * It is checked if lower/upper cases in names of projections are treated properly
#         (should be case letter invariance).




default_par={"projections":None}
input_file="wannier_converter_proj.win"
reporter=AsciiIO()
converter=Wannier90Converter(filename=input_file)

reporter.read_ASCII_fortran_file(file_to_read = input_file, default_dic=default_par)

converter._win_par["projections"]=default_par["projections"]
converter._get_shells() # sort is calculated from HR,
ar = HDFArchive('wannier_converter_test_proj.output.h5', 'w')
ar["shells"]=converter.shells
del ar

# correct_shells=[{'dim': 4, 'l': -3, 'atom': 0}, # As:sp3
#
#                 {'dim': 4, 'l': -3, 'atom': 1}, # Dummy_atom:sp3-1
#                                                 # dUmmy_atom:sp3-2 #  equivalent to As atom (As:sp3)
#                                                 # duMMy_atom:sp3-3
#                                                 # duMMy_Atom:sp3-4
#
#                 {'dim': 5, 'l': 2, 'atom': 2},  # Cu:d
#
#                 {'dim': 5, 'l': 2, 'atom': 3},  # Cu1:dxy:z=0,0,1
#                                                 # CU1:dxz:z=0,0,1
#                                                 # cu1:dyz:z=0,0,1
#                                                 # Cu1:dx2-y2:z=0,0,1
#                                                 # Cu1:dz2:z=0,0,1
#
#                 {'dim': 5, 'l': 2, 'atom': 4},  # Cu2:dxy;dxz;dyz;dx2-y2;dz2:z=0,0,1
#
#                 {'dim': 5, 'l': 2, 'atom': 5},  # Cu3:l=2,mr=2,4
#                                                 # Cu3:l=2,mr=1,3,5 ! bum bum
#
#                 {'dim': 1, 'l': 0, 'atom': 6},  # f=0.25,0.25,0.25:s
#
#                 {'dim': 1, 'l': 0, 'atom': 7},  # f=-0.25,-0.25,-0.25:s # bum bum bum
#
#                 {'dim': 1, 'l': 0, 'atom': 8},  # Fe:s;p;d
#                 {'dim': 3, 'l': 1, 'atom': 8},
#                 {'dim': 5, 'l': 2, 'atom': 8},
#
#                 {'dim': 3, 'l': 1, 'atom': 9},  # O:p
#
#                 {'dim': 3, 'l': 1, 'atom': 10}] # O1:px;py;pz
#

