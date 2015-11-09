from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive import HDFArchive

#    Test for the rearrangement of spinors.
#
#    Input files: wannier_converter_Fe_d.win, wannier_converter_Fe_spd.win, 
#                 wannier_converter_Fe_spd_non_standard.win, wannier_converter_Fe_spd_non_standard_extra.par,
#                 wannier_converter_soc_order.py                  
#
#    Output files: wannier_converter_soc_order.output.h5
#
#    Command: pytriqs wannier_converter_soc_order.py
#
#    Description:
#    The order of spinors used by Wannier90 package is different than the order required by the interface. 
#    In the following benchmark it is verified if the  correct rearrangement of spinors structure
#    for each shell can be obtained so that:
# 
#    U         NON_ZERO
#    NON_ZERO  D  
#
#    Explanation of used symbols:
#
#               U - block up  
#               D - block down
#               NON_ZERO - mixed up/down blocks
#  
#    Once the proper rearrangement is found the content of the Hamiltonian, T matrices and U matrices can
#    be rearranged so that it can be used by the interface.
# 
#    Tested features within this benchmark:
#
#        * rearrangement of spinors in case only one correlated shell is in the system: wannier_converter_Fe_d
#       
#        * rearrangement of spinors in case correlated and non-correlated shells are
#          in the system: wannier_converter_Fe_spd
#
#        * rearrangement of spinors in case few correlated shells are in the system
#          (without non-correlated shells): wannier_converter_Fe_spd_non_standard
#

def benchmarking_fun(dft_filename):
    """

    Benchmarking function.

    :param dft_filename: name of win file to open (dft_filename.win)
    :type dft_filename: str
    """
    Converter = Wannier90Converter(filename=dft_filename)
    Converter._read_win_file()
    for i, state in enumerate(Converter.shells):
        state["sort"] = i # here we fake the sort parameter for the purpose of this benchmark
          
    Converter._set_corr_shells()
    
    # evaluates number of correlated shells
    Converter.n_corr_shells = len(Converter.corr_shells)
    
    # evaluates inequivalent correlated shells and mapping from
    # correlated to non-correlated and from non-correlated to correlated
    Converter.n_inequiv_corr_shells, Converter.corr_to_inequiv, Converter.inequiv_to_corr = Converter.det_shell_equivalence(corr_shells=Converter.corr_shells)    
 
    Converter._set_T()

    # saves data
    ar = HDFArchive('wannier_converter_soc_order.output.h5', 'a')
    if dft_filename not in ar: ar.create_group(dft_filename)
    ar[dft_filename]["win_par"] = Converter._win_par
    ar[dft_filename]["soc_arrangement"] = Converter._soc_arrangement
    ar[dft_filename]["shells"] = Converter.shells
    ar[dft_filename]["corr_shells"] = Converter.corr_shells
    ar[dft_filename]["T"] = Converter.T
    del ar

# Explanation of used symbols:
# u = up 
# d = down 

# Test 1): wannier_converter_Fe_d
# initial order 
# dz2_u dz2_d dxz_u dxz_d d_yz_u d_yz_d dx2-y2_u dx2-y2_d dxy_u d_xy_d

# after rearrangement
# dz2_u dxz_u d_yz_u dx2-y2_u dxy_u dz2_d dxz_d d_yz_d dx2-y2_d d_xy_d

# soc_arrangement
# {'0': 0, '1': 2, '2': 4, '3': 6, '4': 8, '5': 1, '6': 3, '7': 5, '8': 7, '9': 9, }
benchmarking_fun("wannier_converter_Fe_d")



# Test 2): wannier_converter_Fe_spd
# initial order 
# s_u s_d pz_u pz_d px_u px_d py_u py_d dz2_u dz2_d dxz_u dxz_d d_yz_u d_yz_d dx2-y2_u dx2-y2_d dxy_u d_xy_d

# after rearrangement
# s_u s_d pz_u px_u py_u pz_d px_d py_d dz2_u dxz_u d_yz_u dx2-y2_u dxy_u dz2_d dxz_d d_yz_d dx2-y2_d d_xy_d

# soc_arrangement
# {'0': 0, '1': 1, '2': 2, '3': 4, '4': 6, '5': 3, '6': 5, '7': 7, '8': 8, '9': 10, 
#  '10': 12, '11': 14, '12': 16, '13': 9, '14': 11, '15': 13, '16': 15, '17': 17}
benchmarking_fun("wannier_converter_Fe_spd")



# Test 3): wannier_converter_Fe_spd_non_standard (unrealistic example but valid as a benchmark)
# the rearrangement the same as for wannier_converter_Fe_spd
benchmarking_fun("wannier_converter_Fe_spd_non_standard")

