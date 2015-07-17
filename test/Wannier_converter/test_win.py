from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter

default_par={"projections":None}
input_file="LaTiO3" # Core part of the name of file here. File input_file+".win" will be opened

converter=Wannier90Converter(filename=input_file)

# get data from *.win file
converter._read_win_file()

valid_par={'spinors': 'false',
           'hr_plot': 'true',
           'fermi_energy': '15.3695',
           'num_wann': '48',
           'num_bands': '76',
           'mp_grid': '9 9 6',
           'projections':'f=0.000000000,0.000000000,0.000000000:dxy:x=1,1,0\n'
                         'f=0.000000000,0.000000000,0.000000000:dyz:x=1,1,0\n'
                         'f=0.000000000,0.000000000,0.000000000:dxz:x=1,1,0\n'
                         'f=0.500000000,0.500000000,0.500000000:dxy:x=1,1,0\n'
                         'f=0.500000000,0.500000000,0.500000000:dyz:x=1,1,0\n'
                         'f=0.500000000,0.500000000,0.500000000:dxz:x=1,1,0\n'
                         'f=0.500000000,0.500000000,0.000000000:dxy:x=1,1,0\n'
                         'f=0.500000000,0.500000000,0.000000000:dyz:x=1,1,0\n'
                         'f=0.500000000,0.500000000,0.000000000:dxz:x=1,1,0\n'
                         'f=0.000000000,0.000000000,0.500000000:dxy:x=1,1,0\n'
                         'f=0.000000000,0.000000000,0.500000000:dyz:x=1,1,0\n'
                         'f=0.000000000,0.000000000,0.500000000:dxz:x=1,1,0\n'
                         'f=0.075204155,0.016593341,0.250000000:px:x=1,1,0\n'
                         'f=0.075204155,0.016593341,0.250000000:py:x=1,1,0\n'
                         'f=0.075204155,0.016593341,0.250000000:pz:x=1,1,0\n'
                         'f=-0.075204155,-0.016593341,-0.250000000:px:x=1,1,0\n'
                         'f=-0.075204155,-0.016593341,-0.250000000:py:x=1,1,0\n'
                         'f=-0.075204155,-0.016593341,-0.250000000:pz:x=1,1,0\n'
                         'f=0.424794677,0.516543925,0.250000000:px:x=1,1,0\n'
                         'f=0.424794677,0.516543925,0.250000000:py:x=1,1,0\n'
                         'f=0.424794677,0.516543925,0.250000000:pz:x=1,1,0\nf'
                         '=0.575205323,0.483456075,-0.250000000:px:x=1,1,0\n'
                         'f=0.575205323,0.483456075,-0.250000000:py:x=1,1,0\n'
                         'f=0.575205323,0.483456075,-0.250000000:pz:x=1,1,0\n'
                         'f=0.205581716,0.292233800,-0.039745169:px:x=1,1,0\n'
                         'f=0.205581716,0.292233800,-0.039745169:py:x=1,1,0\n'
                         'f=0.205581716,0.292233800,-0.039745169:pz:x=1,1,0\n'
                         'f=0.705581107,0.207760997,0.039728511:px:x=1,1,0\n'
                         'f=0.705581107,0.207760997,0.039728511:py:x=1,1,0\n'
                         'f=0.705581107,0.207760997,0.039728511:pz:x=1,1,0\n'
                         'f=0.294418893,0.792239003,-0.039728511:px:x=1,1,0\n'
                         'f=0.294418893,0.792239003,-0.039728511:py:x=1,1,0\n'
                         'f=0.294418893,0.792239003,-0.039728511:pz:x=1,1,0\n'
                         'f=0.794418284,0.707766200,0.039745169:px:x=1,1,0\n'
                         'f=0.794418284,0.707766200,0.039745169:py:x=1,1,0\n'
                         'f=0.794418284,0.707766200,0.039745169:pz:x=1,1,0\n'
                         'f=0.205581716,0.292233800,0.539745169:px:x=1,1,0\n'
                         'f=0.205581716,0.292233800,0.539745169:py:x=1,1,0\n'
                         'f=0.205581716,0.292233800,0.539745169:pz:x=1,1,0\n'
                         'f=0.705581107,0.207760997,0.460271489:px:x=1,1,0\n'
                         'f=0.705581107,0.207760997,0.460271489:py:x=1,1,0\n'
                         'f=0.705581107,0.207760997,0.460271489:pz:x=1,1,0\n'
                         'f=0.294418893,0.792239003,0.539728511:px:x=1,1,0\n'
                         'f=0.294418893,0.792239003,0.539728511:py:x=1,1,0\n'
                         'f=0.294418893,0.792239003,0.539728511:pz:x=1,1,0\n'
                         'f=0.794418284,0.707766200,0.460254831:px:x=1,1,0\n'
                         'f=0.794418284,0.707766200,0.460254831:py:x=1,1,0\n'
                         'f=0.794418284,0.707766200,0.460254831:pz:x=1,1,0'}

success=True
for item in valid_par:
    try:
        if valid_par[item]!=converter._win_par[item]:
            converter.make_statement(item+ " is invalid!")
            success=False
    except KeyError:
        converter.make_statement(item+ " is invalid!")
        success=False


if success:
    converter.make_statement("Reading data from win file: test passed!")
else: 
    converter.make_statement("Reading data from win file: test failed!")

# postprocess data from *.win file to get converter attributes
converter._get_chemical_potential()
converter._get_ham_nkpt()
converter._get_shells() # no sort calculated here, it is calculated from data in *_hr.dat!
converter._get_total_MLWF()
converter._get_num_bands()

converter_attr={"SO": converter.SO,
                "SP": converter.SP,
                "ham_nkpt": converter.ham_nkpt,
                "shells": converter.shells,
                "total_MLWF": converter.total_MLWF,
                "total_Bloch":converter.total_Bloch}

correct_attr={"SO":0,
              "SP":0,
              "ham_nkpt":[9, 9, 6],
              "shells": [{'dim': 3, 'l': 2, 'atom': 0},
                         {'dim': 3, 'l': 2, 'atom': 1},
                         {'dim': 3, 'l': 2, 'atom': 2},
                         {'dim': 3, 'l': 2, 'atom': 3},
                         {'dim': 3, 'l': 1, 'atom': 4},
                         {'dim': 3, 'l': 1, 'atom': 5},
                         {'dim': 3, 'l': 1, 'atom': 6},
                         {'dim': 3, 'l': 1, 'atom': 7},
                         {'dim': 3, 'l': 1, 'atom': 8},
                         {'dim': 3, 'l': 1, 'atom': 9},
                         {'dim': 3, 'l': 1, 'atom': 10},
                         {'dim': 3, 'l': 1, 'atom': 11},
                         {'dim': 3, 'l': 1, 'atom': 12},
                         {'dim': 3, 'l': 1, 'atom': 13},
                         {'dim': 3, 'l': 1, 'atom': 14},
                         {'dim': 2, 'l': 1, 'atom': 15}], # should be 3
               "total_MLWF": 48,
               "total_Bloch": 76}

success=True
for item in converter_attr:
    try:
        if converter_attr[item]!=converter_attr[item]:
            converter.make_statement(item+ " is invalid!")
            success=False
    except KeyError:
        converter.make_statement(item+ " is invalid!")
        success=False

if success:
    converter.make_statement("Evaluating wannier converter attributes: test passed!")
else:
    converter.make_statement("Evaluating wannier converter attributes: test failed!")