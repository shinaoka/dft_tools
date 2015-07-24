# load triqs helper to set up tests
find_package(TriqsTest)

FILE(COPY SrVO3.h5 SrVO3_Sigma.h5 SrVO3.pmat SrVO3.struct SrVO3.outputs
          SrVO3.oubwin SrVO3.ctqmcout SrVO3.symqmc SrVO3.sympar SrVO3.parproj
          hk_convert_hamiltonian.hk  wannier_converter_test_proj.py
          wannier_converter_proj.win wannier_converter_test_proj.output.h5
          LaTiO3.win wannier_converter_test_win.py wannier_converter_test_win.output.h5
          wannier_converter_SrVO3.win wannier_converter_SrVO3.chk.fmt wannier_converter_test_chk_fmt.py wannier_converter_test_chk_fmt.output.h5
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
triqs_add_test_hdf(wien2k_convert " -p 1.e-6" )
triqs_add_test_hdf(hk_convert " -p 1.e-6" )
triqs_add_test_hdf(sumkdft_basic " -d 1.e-6" )
triqs_add_test_hdf(srvo3_Gloc "  -d 1.e-6" )
triqs_add_test_hdf(srvo3_transp " -d 1.e-6" )
triqs_add_test_hdf(wannier_converter_test_proj " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_test_win " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_test_chk_fmt " -d 1.e-6")
