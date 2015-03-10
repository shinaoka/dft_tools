from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter

default_par={"projections":None}
input_file="proj.win"
reporter=AsciiIO()
converter=Wannier90Converter(filename=input_file)

reporter.read_ASCII_fortran_file(file_to_read = input_file, default_dic=default_par)

converter._win_par["projections"]=default_par["projections"]
converter._get_shells() # sort is calculated from HR,

correct_shells=[{'dim': 4, 'l': -3, 'atom': 0}, # As:sp3

                {'dim': 4, 'l': -3, 'atom': 1}, # Dummy_atom:sp3-1
                                                # dUmmy_atom:sp3-2 #  equivalent to As atom
                                                # duMMy_atom:sp3-3 
                                                # duMMy_Atom:sp3-4

                {'dim': 5, 'l': 2, 'atom': 2},  # Cu:d    

                {'dim': 5, 'l': 2, 'atom': 3},  # Cu1:dxy:z=0,0,1
                                                # CU1:dxz:z=0,0,1
                                                # cu1:dyz:z=0,0,1
                                                # Cu1:dx2-y2:z=0,0,1 
                                                # Cu1:dz2:z=0,0,1

                {'dim': 5, 'l': 2, 'atom': 4},  # Cu2:dxy;dxz;dyz;dx2-y2;dz2:z=0,0,1

                {'dim': 5, 'l': 2, 'atom': 5},  # Cu3:l=2,mr=2,4
                                                # Cu3:l=2,mr=1,3,5 ! bum bum

                {'dim': 1, 'l': 0, 'atom': 6},  # f=0.25,0.25,0.25:s 

                {'dim': 1, 'l': 0, 'atom': 7},  # f=-0.25,-0.25,-0.25:s # bum bum bum

                {'dim': 1, 'l': 0, 'atom': 8},  # Fe:s;p;d
                {'dim': 3, 'l': 1, 'atom': 8}, 
                {'dim': 5, 'l': 2, 'atom': 8}, 

                {'dim': 3, 'l': 1, 'atom': 9},  # O:p

                {'dim': 3, 'l': 1, 'atom': 10}] # O1:px;py;pz

success=True
for n_shell in range(len(converter.shells)):
   if cmp(correct_shells[n_shell],converter.shells[n_shell])!=0:
       converter.make_statement("Test Failed!  %s is different than %s!"%(correct_shells[n_shell],onverter.shells[n_shell]))
       success=False
       break
if success:
    converter.make_statement("Test successful!")

