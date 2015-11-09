from pytriqs.applications.dft.converters.wannier90_converter import Wannier90Converter
from pytriqs.archive.hdf_archive import HDFArchive

#    Test for validity of constructing T matrix for the given order of the initial
#    correlated projections from  *.win file.
#
#    Input files:  wannier_converter_T_d.win, wannier_converter_T_d_long.win, wannier_converter_T_d_other.win,
#                  wannier_converter_T_d_wien2k.win, wannier_converter_T_f.win, wannier_converter_T_order.py
#
#    Output files: wannier_converter_T_order.output.h5
#
#    Command: pytriqs wannier_converter_T_order.py
#
#    Description:
#    In this benchmark a different order of the initial projections in *win file is considered. The different order of
#    projections leads to a different T matrix. In this test it is verified if T is evaluated properly
#    for the given order of the initial projections.
#
#    Tested features within this benchmark:
#
#         *  whether T matrix for d orbitals is evaluated properly
#
#         *  whether T matrix for f orbitals is evaluated properly



def benchmark(filename=None):
    """
     A benchmarking function.
    :param filename: name of win file to open (filename.win)

    """

    # creates an converter object
    converter = Wannier90Converter(filename=filename)

    # reads keywords from filename.win file
    converter._read_win_file()
    
    # adds manually sort to shells
    #
    # In the real calculation  we need *seedname_hr.dat 
    # file in this step.  Whether sort is evaluated properly is tested in another test.
    # Here we have four shells: first shell is always correlated, every shell has a different sort
    # (in general case correlated shell does not have to be the first).
    for i, shell in enumerate(converter.shells):
        shell["sort"] = i

    # evaluates correlated shells
    converter._set_corr_shells()

    # evaluates number of correlated shells
    converter.n_corr_shells = len(converter.corr_shells)
    
    # evaluates inequivalent correlated shells and mapping from
    # correlated to non-correlated and from non-correlated to correlated
    converter.n_inequiv_corr_shells, converter.corr_to_inequiv, converter.inequiv_to_corr = converter.det_shell_equivalence(corr_shells=converter.corr_shells)

    # evaluates T according to the order of initial projections found in *win file
    converter._set_T()

    # shortens name of the directory in hdf file
    folder_name = filename.replace("wannier_converter", "").strip()

    # saves data
    ar = HDFArchive('wannier_converter_T_order.output.h5', 'a')
    if folder_name not in ar: ar.create_group(folder_name)
    ar[folder_name]["win_par"] = converter._win_par
    ar[folder_name]["shells"] = converter.shells
    ar[folder_name]["corr_shells"] = converter.corr_shells
    ar[folder_name]["T"] = converter.T
    del ar


# let's start benchmarking....
benchmark(filename="wannier_converter_T_d")
benchmark(filename="wannier_converter_T_d_long")
benchmark(filename="wannier_converter_T_d_other")
benchmark(filename="wannier_converter_T_d_wien2k")
benchmark(filename="wannier_converter_T_f")



