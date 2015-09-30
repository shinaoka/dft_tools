# load triqs helper to set up tests
find_package(TriqsTest)

FILE(COPY SrVO3.h5 SrVO3_Sigma.h5 SrVO3.pmat SrVO3.struct SrVO3.outputs
          SrVO3.oubwin SrVO3.ctqmcout SrVO3.symqmc SrVO3.sympar SrVO3.parproj
          hk_convert_hamiltonian.hk  wannier_converter_proj.py
          wannier_converter_proj.win wannier_converter_proj.output.h5
          LaTiO3.win wannier_converter_win.py wannier_converter_win.output.h5
          wannier_converter_extra_par.py wannier_converter_extra.par wannier_converter_extra_par.output.h5 
          wannier_converter_SrVO3.win wannier_converter_SrVO3.chk.fmt wannier_converter_chk_fmt.py wannier_converter_chk_fmt.output.h5
          wannier_converter_T_d_long.win  wannier_converter_T_d_other.win  wannier_converter_T_d_wien2k.win  
          wannier_converter_T_d.win  wannier_converter_T_order.output.h5  wannier_converter_T_order.py  wannier_converter_T_f.win
          wannier_converter_SrVO3_hr.dat wannier_converter_chk_fmt_shells.output.h5 wannier_converter_chk_fmt_shells.py
          wannier_converter_dft_dmft_cthyb_extract_G_loc.py wannier_converter_dft_dmft_cthyb_extract_G_loc.output.h5
          wannier_converter_dft_dmft_cthyb_chem_pot.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
triqs_add_test_hdf(wien2k_convert " -p 1.e-6" )
triqs_add_test_hdf(hk_convert " -p 1.e-6" )
triqs_add_test_hdf(sumkdft_basic " -d 1.e-6" )
triqs_add_test_hdf(srvo3_Gloc "  -d 1.e-6" )
triqs_add_test_hdf(srvo3_transp " -d 1.e-6" )
triqs_add_test_hdf(wannier_converter_extra_par " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_proj " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_win " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_chk_fmt " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_chk_fmt_shells " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_T_order " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_dft_dmft_cthyb_chem_pot " -d 1.e-6")
triqs_add_test_hdf(wannier_converter_dft_dmft_cthyb_extract_G_loc " -d 1.e-6")
