from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive import HDFArchive

# Test for validity of reading  initial guesses from *.win file in the simplified format.
#
#    Input files: LaTiO3_atom_name.win, LaTiO3.win, wannier_converter_proj_by_atom_name.py
#
#    Output files:  wannier_converter_proj_by_atom_name.output.h5
#
#    Command: pytriqs wannier_converter_proj_by_atom_name.py
#
#    Description:
#    In LaTiO3_atom_name.win file hypothetical initial guesses in the simplify format are stored. In file LaTiO3.win
#    equivalent initial guesses but in the longer format are stored. In this test it is verified whether Wannier90
#    converter can properly transform blocks:
#
#    begin atoms_frac
#    ----------------
#    ----------------
#    end atoms_frac
#     
#    "------"  symbolizes valid entry with name of atom and correesponding coordinates    
#
#    and
#  
#    begin projections
#    -----------------
#    -----------------
#
#    end projections
# 
#    "------"  symbolizes valid entry with projection (or projections) written in the format accepted by Wannier90.
#    
#    to shells.
#
#    Tested features within this benchmark:
#
#       * Reconstruction of shells basing only on names of atoms, atoms_frac  and initial guesses for these atoms.
#         In this test it is verified if longer version:
#
#           begin projections
#           f=0.000000000,0.000000000,0.000000000:dxy:x=1,1,0
#           f=0.000000000,0.000000000,0.000000000:dyz:x=1,1,0
#           f=0.000000000,0.000000000,0.000000000:dxz:x=1,1,0
#           f=0.500000000,0.500000000,0.500000000:dxy:x=1,1,0
#           f=0.500000000,0.500000000,0.500000000:dyz:x=1,1,0
#           f=0.500000000,0.500000000,0.500000000:dxz:x=1,1,0
#           f=0.500000000,0.500000000,0.000000000:dxy:x=1,1,0
#           f=0.500000000,0.500000000,0.000000000:dyz:x=1,1,0
#           f=0.500000000,0.500000000,0.000000000:dxz:x=1,1,0
#           f=0.000000000,0.000000000,0.500000000:dxy:x=1,1,0
#           f=0.000000000,0.000000000,0.500000000:dyz:x=1,1,0
#           f=0.000000000,0.000000000,0.500000000:dxz:x=1,1,0
#           f=0.075204155,0.016593341,0.250000000:pz:x=1,1,0
#           f=0.075204155,0.016593341,0.250000000:px:x=1,1,0
#           f=0.075204155,0.016593341,0.250000000:py:x=1,1,0
#           f=-0.075204155,-0.016593341,-0.250000000:pz:x=1,1,0
#           f=-0.075204155,-0.016593341,-0.250000000:px:x=1,1,0
#           f=-0.075204155,-0.016593341,-0.250000000:py:x=1,1,0
#           f=0.424794677,0.516543925,0.250000000:pz:x=1,1,0
#           f=0.424794677,0.516543925,0.250000000:px:x=1,1,0
#           f=0.424794677,0.516543925,0.250000000:py:x=1,1,0
#           f=0.575205323,0.483456075,-0.250000000:pz:x=1,1,0
#           f=0.575205323,0.483456075,-0.250000000:px:x=1,1,0
#           f=0.575205323,0.483456075,-0.250000000:py:x=1,1,0
#           f=0.205581716,0.292233800,-0.039745169:pz:x=1,1,0
#           f=0.205581716,0.292233800,-0.039745169:px:x=1,1,0
#           f=0.205581716,0.292233800,-0.039745169:py:x=1,1,0
#           f=0.705581107,0.207760997,0.039728511:pz:x=1,1,0
#           f=0.705581107,0.207760997,0.039728511:px:x=1,1,0
#           f=0.705581107,0.207760997,0.039728511:py:x=1,1,0
#           f=0.294418893,0.792239003,-0.039728511:pz:x=1,1,0
#           f=0.294418893,0.792239003,-0.039728511:px:x=1,1,0
#           f=0.294418893,0.792239003,-0.039728511:py:x=1,1,0
#           f=0.794418284,0.707766200,0.039745169:pz:x=1,1,0
#           f=0.794418284,0.707766200,0.039745169:px:x=1,1,0
#           f=0.794418284,0.707766200,0.039745169:py:x=1,1,0
#           f=0.205581716,0.292233800,0.539745169:pz:x=1,1,0
#           f=0.205581716,0.292233800,0.539745169:px:x=1,1,0
#           f=0.205581716,0.292233800,0.539745169:py:x=1,1,0
#           f=0.705581107,0.207760997,0.460271489:pz:x=1,1,0
#           f=0.705581107,0.207760997,0.460271489:px:x=1,1,0
#           f=0.705581107,0.207760997,0.460271489:py:x=1,1,0
#           f=0.294418893,0.792239003,0.539728511:pz:x=1,1,0
#           f=0.294418893,0.792239003,0.539728511:px:x=1,1,0
#           f=0.294418893,0.792239003,0.539728511:py:x=1,1,0
#           f=0.794418284,0.707766200,0.460254831:pz:x=1,1,0
#           f=0.794418284,0.707766200,0.460254831:px:x=1,1,0
#           f=0.794418284,0.707766200,0.460254831:py:x=1,1,0
#           end projections
#
#           is properly substituted by a shorted version:
#
#           Begin Projections
#           Ti : dxy; dyz; dxz:x=1,1,0
#           O  : p:x=1,1,0
#           End Projections


def test_me(dft_filename):
	"""
	Function for testing.
	:param dft_filename: name of win file (dft_filename.win)
	"""
	Converter = Wannier90Converter(filename=dft_filename)
	Converter._read_win_file()
	ar = HDFArchive('wannier_converter_proj_by_atom_name.output.h5', 'a')
	ar[dft_filename+"-win_par"] = Converter._win_par
	ar[dft_filename+"-shells"] = Converter.shells
	del ar


files = ['LaTiO3_atom_name', "LaTiO3"]

for f in files:
	test_me(dft_filename=f)




