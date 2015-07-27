from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive.hdf_archive import  HDFArchive

#    Test for validity of read parameters from *.win file
#
#    Input files: LaTiO3.win, wannier_converter_test_win.py
#
#    Output files: wannier_converter_test_win.output.h5
#
#    Command: pytriqs wannier_converter_test_win.py
#
#    Description:
#    In this benchmark a realistic case of LaTiO3 with Ti t2g and  O 2p states in the local Hamiltonian is considered.
#    It is verified if data is read properly from LaTiO3.win file (all content of win file is provided here directly
#    from wannier90 calculations). Win file is an input file for wannier90.
#
#    Tested features within this benchmark:
#
#        * It is checked if lower/upper cases of letters are treated as the same (Fortran style).
#
#        * Various ways of defining keyword-value pair are used:
#
#                num_wann 4
#
#                num_wann = 4
#
#                num_wann : 4
#
#           and it is checked if in all cases  keywords are read properly.
#
#        * It is tested if attributes of wannier90 converter are extracted properly
#          from data which was read from LaTiO3.win file.

input_file="LaTiO3" # Core part of the name of file here. File input_file+".win" will be opened

converter=Wannier90Converter(filename=input_file)

# get data from *.win file
converter._read_win_file()

# extract converrter attributes
converter_attr={"SO": converter.SO,
                "SP": converter.SP,
                "ham_nkpt": converter.ham_nkpt,
                "shells": converter.shells, # sort calculater in other benchmark!
                "total_MLWF": converter.total_MLWF,
                "total_Bloch":converter.total_Bloch,
                "chemical_potential":converter.chemical_potential}

ar = HDFArchive('wannier_converter_test_win.output.h5', 'w')
ar["win_par"]=converter._win_par
ar["converter_attr"]=converter_attr
del ar

# valid_par={'spinors': 'false',
#            'hr_plot': 'true',
#            'fermi_energy': '15.3695',
#            'num_wann': '48',
#            'num_bands': '76',
#            'mp_grid': '9 9 6',
#            'projections':'f=0.000000000,0.000000000,0.000000000:dxy:x=1,1,0\n'
#                          'f=0.000000000,0.000000000,0.000000000:dyz:x=1,1,0\n'
#                          'f=0.000000000,0.000000000,0.000000000:dxz:x=1,1,0\n'
#                          'f=0.500000000,0.500000000,0.500000000:dxy:x=1,1,0\n'
#                          'f=0.500000000,0.500000000,0.500000000:dyz:x=1,1,0\n'
#                          'f=0.500000000,0.500000000,0.500000000:dxz:x=1,1,0\n'
#                          'f=0.500000000,0.500000000,0.000000000:dxy:x=1,1,0\n'
#                          'f=0.500000000,0.500000000,0.000000000:dyz:x=1,1,0\n'
#                          'f=0.500000000,0.500000000,0.000000000:dxz:x=1,1,0\n'
#                          'f=0.000000000,0.000000000,0.500000000:dxy:x=1,1,0\n'
#                          'f=0.000000000,0.000000000,0.500000000:dyz:x=1,1,0\n'
#                          'f=0.000000000,0.000000000,0.500000000:dxz:x=1,1,0\n'
#                          'f=0.075204155,0.016593341,0.250000000:px:x=1,1,0\n'
#                          'f=0.075204155,0.016593341,0.250000000:py:x=1,1,0\n'
#                          'f=0.075204155,0.016593341,0.250000000:pz:x=1,1,0\n'
#                          'f=-0.075204155,-0.016593341,-0.250000000:px:x=1,1,0\n'
#                          'f=-0.075204155,-0.016593341,-0.250000000:py:x=1,1,0\n'
#                          'f=-0.075204155,-0.016593341,-0.250000000:pz:x=1,1,0\n'
#                          'f=0.424794677,0.516543925,0.250000000:px:x=1,1,0\n'
#                          'f=0.424794677,0.516543925,0.250000000:py:x=1,1,0\n'
#                          'f=0.424794677,0.516543925,0.250000000:pz:x=1,1,0\nf'
#                          '=0.575205323,0.483456075,-0.250000000:px:x=1,1,0\n'
#                          'f=0.575205323,0.483456075,-0.250000000:py:x=1,1,0\n'
#                          'f=0.575205323,0.483456075,-0.250000000:pz:x=1,1,0\n'
#                          'f=0.205581716,0.292233800,-0.039745169:px:x=1,1,0\n'
#                          'f=0.205581716,0.292233800,-0.039745169:py:x=1,1,0\n'
#                          'f=0.205581716,0.292233800,-0.039745169:pz:x=1,1,0\n'
#                          'f=0.705581107,0.207760997,0.039728511:px:x=1,1,0\n'
#                          'f=0.705581107,0.207760997,0.039728511:py:x=1,1,0\n'
#                          'f=0.705581107,0.207760997,0.039728511:pz:x=1,1,0\n'
#                          'f=0.294418893,0.792239003,-0.039728511:px:x=1,1,0\n'
#                          'f=0.294418893,0.792239003,-0.039728511:py:x=1,1,0\n'
#                          'f=0.294418893,0.792239003,-0.039728511:pz:x=1,1,0\n'
#                          'f=0.794418284,0.707766200,0.039745169:px:x=1,1,0\n'
#                          'f=0.794418284,0.707766200,0.039745169:py:x=1,1,0\n'
#                          'f=0.794418284,0.707766200,0.039745169:pz:x=1,1,0\n'
#                          'f=0.205581716,0.292233800,0.539745169:px:x=1,1,0\n'
#                          'f=0.205581716,0.292233800,0.539745169:py:x=1,1,0\n'
#                          'f=0.205581716,0.292233800,0.539745169:pz:x=1,1,0\n'
#                          'f=0.705581107,0.207760997,0.460271489:px:x=1,1,0\n'
#                          'f=0.705581107,0.207760997,0.460271489:py:x=1,1,0\n'
#                          'f=0.705581107,0.207760997,0.460271489:pz:x=1,1,0\n'
#                          'f=0.294418893,0.792239003,0.539728511:px:x=1,1,0\n'
#                          'f=0.294418893,0.792239003,0.539728511:py:x=1,1,0\n'
#                          'f=0.294418893,0.792239003,0.539728511:pz:x=1,1,0\n'
#                          'f=0.794418284,0.707766200,0.460254831:px:x=1,1,0\n'
#                          'f=0.794418284,0.707766200,0.460254831:py:x=1,1,0\n'
#                          'f=0.794418284,0.707766200,0.460254831:pz:x=1,1,0'}
#
#
# converter_attr={"SO": converter.SO,
#                 "SP": converter.SP,
#                 "ham_nkpt": converter.ham_nkpt,
#                 "shells": converter.shells,
#                 "total_MLWF": converter.total_MLWF,
#                 "total_Bloch":converter.total_Bloch,
#                 "chemical_potential":converter.chemical_potential }
#
# correct_attr={"SO":0,
#               "SP":0,
#               "ham_nkpt":[9, 9, 6],
#               "shells": [{'dim': 3, 'l': 2, 'atom': 0},
#                          {'dim': 3, 'l': 2, 'atom': 1},
#                          {'dim': 3, 'l': 2, 'atom': 2},
#                          {'dim': 3, 'l': 2, 'atom': 3},
#                          {'dim': 3, 'l': 1, 'atom': 4},
#                          {'dim': 3, 'l': 1, 'atom': 5},
#                          {'dim': 3, 'l': 1, 'atom': 6},
#                          {'dim': 3, 'l': 1, 'atom': 7},
#                          {'dim': 3, 'l': 1, 'atom': 8},
#                          {'dim': 3, 'l': 1, 'atom': 9},
#                          {'dim': 3, 'l': 1, 'atom': 10},
#                          {'dim': 3, 'l': 1, 'atom': 11},
#                          {'dim': 3, 'l': 1, 'atom': 12},
#                          {'dim': 3, 'l': 1, 'atom': 13},
#                          {'dim': 3, 'l': 1, 'atom': 14},
#                          {'dim': 3, 'l': 1, 'atom': 15}],
#                "total_MLWF": 48,
#                "total_Bloch": 76,
#                "chemical_potential": 15.3695}
#
