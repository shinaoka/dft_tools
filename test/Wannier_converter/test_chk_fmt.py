from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO
from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter

import numpy

input_file="SrVO3" # checkpoint is stored in SrVO3.chk.fmt


converter=Wannier90Converter(filename=input_file)

# get data from *.win file
converter._read_win_file()

# get data from *.chk_fmt file
converter_chk_file=converter._read_chkpt_fmt(filename=input_file)

correct_chk_file={'checkpoint': 'postwann',
                  'Have disentanglement': False,
                  'Number of bands': 3,
                  'Number of exclude bands': 27,
                  'Number of Wannier orbitals': 3,
                  'M-P grid': [8, 8, 8],
                  'Exclude_bands': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30],
                  'Reciprocal lattice': numpy.array([[ 1.63535562,  0.        ,  0.        ],
                                               [ 0.        ,  1.63535562,  0.        ],
                                               [ 0.        ,  0.        ,  1.63535562]]),
                  'Number of kpoints': 512,
                  'Real lattice': numpy.array([[ 3.84209112,  0.        ,  0.        ],
                                         [ 0.        ,  3.84209112,  0.        ],
                                         [ 0.        ,  0.        ,  3.84209112]])}

success=True
for entry in correct_chk_file:
   if isinstance(correct_chk_file[entry], numpy.ndarray):
       if not ( correct_chk_file[entry].shape==converter_chk_file[entry].shape and numpy.allclose(correct_chk_file[entry], converter_chk_file[entry])):
           converter.make_statement("Test Failed!  %s is different than %s!"%(correct_chk_file[entry],converter_chk_file[entry]))
           success=False
           break

   else:
       if cmp(correct_chk_file[entry], converter_chk_file[entry])!=0:
           converter.make_statement("Test Failed!  %s is different than %s!"%(correct_chk_file[entry],converter_chk_file[entry]))
           success=False
           break
if success:
    converter.make_statement("Test successful!")

