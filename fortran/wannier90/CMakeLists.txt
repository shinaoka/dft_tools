triqs_build_f2py_module(triqs_wannier90_read_chkpt wannier90_read_chkpt wannier90_read_chkpt.pyf wannier90_read_chkpt.f90)
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/wannier90_read_chkpt.so DESTINATION ${TRIQS_PYTHON_LIB_DEST}/applications/dft/converters)

