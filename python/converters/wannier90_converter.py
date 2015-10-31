"""
Wannier90Converter -- main file.
"""
import numpy
import math
from os.path import isfile
from copy import deepcopy

# pytriqs
from pytriqs.utility.mpi import bcast, is_master_node, slice_array, all_reduce, barrier, world, rank
from pytriqs.archive import HDFArchive
from pytriqs.applications.dft.U_matrix import spherical_to_cubic
from pytriqs.applications.dft.converters.converter_tools import ConverterTools
from pytriqs.applications.dft.messaging import Check, Save, Readh5file
from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO


class Wannier90Converter(Check, ConverterTools, Readh5file, Save):
    """
        **Functionality**::

            (main methods mentioned only, please have a look below for all methods)

            -  Reads H_R from Wannier90.x output file, Fourier transforms it to H_k.
                (method: __h_to_triqs)

            -  Reads projections from seedname*.chk.fmt file(s) and transforms it to the proper format.
               seedname*.chk.fmt are formatted checkpoints files,
               (formatted files store data in a machine independent way)
                (method: _read_chkpt_fmt)

            -  Calculates hopping integrals
                (method: _produce_hopping)

            -  Calculates symmetry operations which transform each inequivalent site
               to all other sites related to that site by symmetry
                (method: __h_to_triqs)

            -  Prepares parameters necessary for SumkDFT
                (method: _SumkDFT_par)

            -  Writes input data for SumkDFT into hdf file
                (inherited method: method _save_par_hdf)

            -  Writes Wannier90Converter parameters into hdf file
                (inherited method: method _save_par_hdf)

            -  Checks if input parameters for SumkDFT have changed since the last run

                (inherited method: parameters_changed)

            -  Checks if input parameters for Wannier90Converter have changed since the last run
            `
                (inherited method: parameters_changed)


        Number of files needed by the converter and their names depend on the polarisation scenario:

            -   Non-polarised calculation:

                   * seedname.win (obligatory),

                   * seedname.chk.fmt (optional),

                   * seedname_hr.dat (obligatory).

            -   Spin-polarised calculation:

                    * seedname_up.win, seedname_down.win (obligatory)

                For each spin channel a calculation is performed
                separately within Wannier90 and as a result we get the following files:

                    * seedname_up.chk.fmt, seedname_down.chk.fmt, (optional),

                    * seedname_hr_up.dat, seedname_hr_down.dat (obligatory).

            -   Spin-orbit coupling  calculation:

                    * seedname.win (obligatory),

                    * seedname.chk.fmt (optional),

                    * seedname_hr.dat (obligatory).

                Keyword spinors = true in seedname.win is obligatory in this case  and
                only half of the initial projections should be provided.


        Expected behaviour in case of non spin-polarised calculations:

            - seedname.win:

                * If there is no seedname.win file a calculation will not start.

            - seedname_hr.dat:

                * If there is no seedname_hr.dat file a calculation will not start.

            - seedname.chk.fmt:

                * If seedname.chk.fmt is not present, dummy projections in the
                  form of rectangular matrices, each with a block of the identity matrix
                  which corresponds to the given correlated shell is calculated.

                * If a number of shells is larger than a number of correlated shells  then seedname.chk.fmt
                  is neglected and dummy projections are always built.

                * Otherwise projections basing on data from  seedname.chk.fmt are constructed.

                * In case seedname.chk.fmt is present a check for consistency
                  between seedname.win and seedname.chk.fmt is done.


        Expected behaviour in case of a spin-orbit calculations:

            - seedname.win:

                * If there is no seedname.win file a calculation will not start.

            - seedname_hr.dat:

                * If there is no seedname_hr.dat file calculation will not start.

            - Whether or not we have spin-orbit calculations is determined basing on a value spinors
              from seedname.win. If there is not such a value then it is assumed that we don't have spin-orbit
              calculations.


        Expected behaviour in case of a spin-polarised calculations:

            - seedname_up.win, seedname_down.win:

                * If there is no seedname_up.win file or seedname_down.win file calculation will not start.

                * Values found in both win files are compared. The only value which can be different
                  is a number of Bloch states used to construct H_R (keyword num_bands in a win file).

            - seedname_up_hr.dat, seedname_down_hr.dat:

                * If there is no seedname_up_hr.dat file or seedname_down_hr.dat calculation will not start.

                * Files seedname_up_hr.dat and seedname_down_hr.dat are compared:
                  nrpt, total_MLWF, Vector_R_degeneracy, Vector_R must be the same for each spin channel.

            - seedname_up.chk.fmt, seedname_down.chk.fmt:

                * In case of spin-polarised calculations if any of seedname*.chk.fmt
                  files is not present then dummy projectors are constructed.

                * Input data from win files is compared against data from  checkpoint files. If there is any
                  discrepancy an abort signal is sent.


        In case dummy projections are built, H_k becomes hopping integrals.

        Spin-orbit calculations: status
            As it comes to spin-orbit coupling at the moment only the case, in which there is the same number
            of spinors up and down, is supported. Usage of brackets in the definition of spinors
            is not supported. This means that projections defined in the following way:

            Fe:dxy;dxz;dyz

            are valid. However, the following format is not valid for the converter:

            Fe:dxy;dxz;dyz(u,d).

            Moreover, only the case of a basis without "mixing" is implemented.

    """

    # static variables:
    corr_shells_keywords = ["atom", "sort", "l", "dim", "SO", "irep"]
    shells_keywords = ["atom", "sort", "l", "dim"]


    def __init__(self, filename=None, dft_subgrp="SumK_DFT", repacking=False):

        """Constructor for Wannier90Converter object.

            :param filename:  			core name of important files: filename_hr.dat,
                                                                      filename.h5,
                                                                      filename.win,
                                                                      filename_up.win,
                                                                      filename_down.win,
                                                                      filename.chk.fmt,
                                                                      filename_up.chk.fmt,
                                                                      filename_down.chk.fmt
                                                                      filename_extra.par
            :type filename: 			str

            :param dft_subgrp:			Name of the directory in filename.h5 where input
                                        parameters for SumkDFT  are stored.
            :type dft_subgrp:			str

            :param repacking:           True if repacking of a hdf file should be done, and False otherwise.
            :type repacking:            bool
            """

        super(Wannier90Converter, self).__init__()

        # ********************* fields ********************************

        self.FULL_H_R = None  # H_R

        self.H_k = None  # H_k

        self.SO = None  # defines whether or not we have a spin-orbit calculation

        self.SP = None  # defines whether or not we have a spin-polarised  calculation

        self.T = None  # Rotation matrices: complex harmonics to cubic harmonics for each inequivalent correlated shell

        self.Vector_R_degeneracy = None  # degeneracy of each Wigner-Seitz grid point

        self.Vector_R = None  # vector with Wigner-Seitz grid points for Fourier transformation H_R -> H_k

        self.chemical_potential = None  # initial guess for the chemical potential

        self.corr_shells = None  # all correlated shells (also symmetry equivalent)

        self.corr_to_inequiv = None  # mapping: correlated shell -> inequivalent correlated shell

        self.density_required = None  # total number of electrons included in the local Hamiltonian

        self.ham_nkpt = None  # vector with three elements which defines k-point grid

        self.hdf_file = None    # name of the hdf file where input data for DMFT is stored.

        self.inequiv_to_corr = None  # mapping: inequivalent correlated shell -> correlated shell

        self.k_dep_projection = None  # whether or not the dimension of projections depend on k-points

        self.k_point_mesh = None  # k-point grid

        self.n_corr_shells = None  # number of all correlated shells (both inequivalent and equivalent)

        self.n_k = None  # total number of k-points

        self.n_inequiv_corr_shells = None  # number of inequivalent correlated shells

        self.n_orbitals = None  # number of bands used to produce MLWFs for each spin channel and each k-point

        self.n_shells = None  # number of all shells (both correlated and non-correlated)

        self.proj_mat = None  # projections for DMFT calculations

        self.rot_mat = None  # rotation matrices for symmetry equivalent sites

        self.shells = None  # all shells (also symmetry equivalent)

        self.total_Bloch = None  # total number of Bloch states used for the construction of MLWFs
                                 # (in case of spin-polarised calculations it is maximum from both spin channels)

        self.total_MLWF = None  # total number of MLWFs (both correlated and non-correlated) in case
                                # of spin-polarised calculation it is expected that the same
                                # number of MLWFs is constructed for each spin channel

        self._R_sym = None  # symmetry operations


        # _all states: data structure with states implemented
        # in Wannier 90 which are used as initial guesses for MLWFs.
        # Each entry is the list with  the following entries:
        #
        #       l --  angular momentum, (but for hybridisation it is just  a label with a negative value)
        #
        #       mr -- number of a state for the particular l
        #
        #       name -- label of a state or group of states
        #
        #       dim -- number of  orbitals for the particular state
        #              (for example for p it is 3, for d it is 5, for dxy it is 1)
        #

        # labels s,p, etc. for states taken from Wannier90 user guide
        # we want to project on orbitals with pure correlated or non-correlated character,
        # orbitals like sp3d are not valid for Wannier90Converter

        self._all_states = [
                            {"l": 0, "mr": 1, "name": "s", "dim": 1},

                            {"l": 1, "mr": None, "name": "p", "dim": 3},
                            {"l": 1, "mr": 1, "name": "pz", "dim": 1},
                            {"l": 1, "mr": 2, "name": "px", "dim": 1},
                            {"l": 1, "mr": 3, "name": "py", "dim": 1},

                            {"l": 2, "mr": None, "name": "d", "dim": 5},
                            {"l": 2, "mr": 1, "name": "dz2", "dim": 1},
                            {"l": 2, "mr": 2, "name": "dxz", "dim": 1},
                            {"l": 2, "mr": 3, "name": "dyz", "dim": 1},
                            {"l": 2, "mr": 4, "name": "dx2-y2", "dim": 1},
                            {"l": 2, "mr": 5, "name": "dxy", "dim": 1},

                            {"l": 3, "mr": None, "name": "f", "dim": 7},
                            {"l": 3, "mr": 1, "name": "fz3", "dim": 1},
                            {"l": 3, "mr": 2, "name": "fxz2", "dim": 1},
                            {"l": 3, "mr": 3, "name": "fyz2", "dim": 1},
                            {"l": 3, "mr": 4, "name": "fxyz", "dim": 1},
                            {"l": 3, "mr": 5, "name": "fz(x2-y2)", "dim": 1},
                            {"l": 3, "mr": 6, "name": "fx(x2-3y2)", "dim": 1},
                            {"l": 3, "mr": 7, "name": "fy(3x2-y2)", "dim": 1},

                            {"l": -1, "mr": None, "name": "sp", "dim": 2},
                            {"l": -1, "mr": 1, "name": "sp-1", "dim": 1},
                            {"l": -1, "mr": 2, "name": "sp-2", "dim": 1},

                            {"l": -2, "mr": None, "name": "sp2", "dim": 3},
                            {"l": -2, "mr": 1, "name": "sp2-1", "dim": 1},
                            {"l": -2, "mr": 2, "name": "sp2-2", "dim": 1},
                            {"l": -2, "mr": 3, "name": "sp2-3", "dim": 1},

                            {"l": -3, "mr": None, "name": "sp3", "dim": 4},
                            {"l": -3, "mr": 1, "name": "sp3-1", "dim": 1},
                            {"l": -3, "mr": 2, "name": "sp3-2", "dim": 1},
                            {"l": -3, "mr": 3, "name": "sp3-3", "dim": 1},
                            {"l": -3, "mr": 4, "name": "sp3-4", "dim": 1}
                           ]

        self._all_read = None  # whether or not data needed for rerun is in the hdf file

        self._atoms = None  # stores all initial projections read so far from win file

        self._disentanglement_spin = None  # defines disentanglement separately for each spin

        self._dummy_projections_used = False  # defines whether or not dummy projections are in use

        self._extra_par = {"num_zero": 0.01,                 # default values of additional parameters for
                           "verbosity": 2,                     # Wannier90Converter. They can be modified
                           "non_standard_corr_shells": False}  # by the user in case additional ASCII file is provided.

        if isinstance(filename, str):
            self._filename = filename
        else:
            self.report_error("Give a string for keyword 'filename' so that input "
                              "Hamiltonian can be read from filename_hr.dat and "
                              "DMFT calculations can be saved into filename.h5!")

        # self._extra_par_file:  name of a file with some additional parameters (num_zero, verbosity, user
        # defined correlated shells)

        self._extra_par_file = self._filename + "_extra.par"

        self.hdf_file = self._filename + ".h5"

        self._name2SpinChannel = {self._filename + "_up": 0,  # defines mapping between name of file and spin channel
                                  self._filename + "_down": 1,
                                  self._filename: 0}

        self._cell_cart = None  # cell lattice vectors in Cartesian coordinates in Angstrom units

        self._num_zero = None  # definition of a numerical zero

        self._n_spin = None  # defines a spin dimension for DFT data,
        #                     it is 1 for a spin non-polarised, spin-orbit calculation
        #                     and 2 for a spin-polarised calculation

        self._parameters = {}  # parameters specific to Wannier90Converter

        # self._SumkDFT:  folder in hdf file where dft input data is stored (both SumkDFT and Wannier90Converter)

        if isinstance(dft_subgrp, str):
            self._SumkDFT = dft_subgrp
        else:
            self.report_error(""""Give a name for a folder in hdf5 in
                              which results from SumkDFT will be kept!""")

        self._total_Bloch_spin = None  # total number of Bloch states used for the construction of MLWFs
                                       # for each spin channel

        self._u_matrix = None  # transformation from (pseudo)-Bloch smooth states to MLWFs

        self._u_matrix_full = None  # transformation form Bloch to MLWFs

        self._u_matrix_opt = None  # transformation form Bloch to pseudo-Bloch smooth states

        self._non_standard_corr_shells = None  # stores user provided correlated shells if such exist

        self._verbosity = None  # defines level of output verbosity

        self._wannier_order = None  # defines the order of orbitals for each shell

        self._wannier_order_corr = None  # defines the order of orbitals for each correlated shell

        self._win_par = {"fermi_energy": None,  # definition of parameters from win file needed by the converter
                         "mp_grid": None,
                         "hr_plot": None,
                         "spinors": "false",
                         "projections": None,
                         # unit_cell_cart keyword is to check compatibility between checkpoint file and win file
                         "unit_cell_cart": None,
                         "num_wann": None,
                         "num_bands": -1}

        # Repacks hdf file  if wanted:
        if isinstance(repacking, bool):
            if repacking:
                self.repack()  # taken from ConverterTools
        else:
            self.report_error("Invalid value for 'repacking' keyword. Bool value is expected!")


        # Gets extra parameters
        self._read_extra_par()


    def convert_dft_input(self):
        """
        Reads the input files, and stores the data in the HDF file. Checks whether we have the rerun or the fresh
        calculation. In case of the rerun checks if parameters have changed. If crucial parameters have changed it
        will abort.
        """

        self._all_read = False
        wann_things_to_check = ["n_k", "SP", "SO", "density_required", "shells", "n_shells",
                                "corr_shells", "n_corr_shells", "T", "n_orbitals", "proj_mat",
                                "hopping", "n_inequiv_shells", "corr_to_inequiv",
                                "inequiv_to_corr", "rot_mat", "energy_unit",
                                "charge_below", "symm_op", "k_dep_projection",
                                "use_rotations", "rot_mat_time_inv", "n_reps", "dim_reps"]

        # here we just check if there is an entry, validation of entry is done later
        self._all_read = self._read_parameters_from_h5_file(subgrp=self._SumkDFT, things_to_load=wann_things_to_check)

        self._read_win_file()

        # parameters needed by both rerun and fresh run
        self._parameters = {"num_zero": self._num_zero,
                            "verbosity": self._verbosity,
                            "ham_nkpt": self.ham_nkpt}

        self._h_to_triqs()
        self._get_density_required()

        if not self._all_read:  # calculation from scratch

            # Deletes a content of the hdf file. Since some parameters were not found
            # it is invalid and starts to  write file from scratch.

            if is_master_node():
                ar = HDFArchive(self.hdf_file, "w")
                ar.create_group(self._SumkDFT)
                del ar

            self._get_T()
            self._produce_projections()

            # bcasting
            self.proj_mat = bcast(self.proj_mat)
            self.n_orbitals = bcast(self.n_orbitals)
            self.total_Bloch = bcast(self.total_Bloch)
            self._dummy_projections_used = bcast(self._dummy_projections_used)
            self._u_matrix_full = bcast(self._u_matrix_full)

            self._produce_hopping()
            self._SumkDFT_par()


            if is_master_node():

                self._save_par_hdf(name=self._SumkDFT,
                                   dictionary=self._parameters)

                self._save_par_hdf(name=self._SumkDFT,
                                   dictionary=self._SumkDFT_data)

            self.make_statement("Data for SumkDFT has been saved to the hdf file.")

        else:  # case of rerun

            # critical parameters, if they change between reruns program should stop

            self._set_k_dep_projection()

            dft_data = {"energy_unit": 1.0,
                        "SP": self.SP,
                        "SO": self.SO,
                        "k_dep_projection": self.k_dep_projection,
                        "charge_below": 0,
                        "symm_op": 0,
                        "shells": self.shells,
                        "n_shells": self.n_shells,
                        "corr_shells": self.corr_shells,
                        "n_corr_shells": len(self.corr_shells),
                        "density_required": self.density_required,
                        "n_k": self.n_k,
                        "use_rotations": 1,
                        "rot_mat": self.rot_mat,
                        "bz_weights": numpy.ones(self.n_k, numpy.float) / float(self.n_k),
                        "rot_mat_time_inv": [0 for i in range(self.n_shells)],
                        "n_reps": -1,
                        "dim_reps": -1,
                        "FULL_H_R": self.FULL_H_R,
                        "Hamiltonian": self.H_k,
                        "ham_nkpt": self.ham_nkpt  # converter crucial parameter
                       }

            # non-crucial parameters; they may change between reruns
            non_crucial_par = {"verbosity": self._verbosity, "num_zero": self._num_zero}

            self._critical_par = dft_data.keys()
            self._critical_par_changed = False

            # Stops if crucial input  parameters for SumkDFT or
            # Wannier90Converter have changed their values since the last rerun.
            self.check_parameters_changed(dictionary=dft_data,
                                          hdf_dir=self._SumkDFT)

            # Warns if non-crucial parameters changed.
            self.check_parameters_changed(dictionary=non_crucial_par,
                                          hdf_dir=self._SumkDFT)

            self.make_statement("Calculation has been restarted.")

        barrier()


    def _read_extra_par(self):
        """
        Reads additional parameters from seedname_extra.par. Case of each letter in every keyword is important.
        In case there is no such a file default values will be used. In case a redefinition of a keyword occurs it
        sends an abort signal. In case of an unknown keyword it sends an abort signal.

        Parameters read from the seedname_extra.par file:

            - verbosity: if verbosity=2 then the additional information  is printed
                         out to the standard output. In particular, the additional information
                         will be printed in case input parameters have changed from the last run.
              verbosity type : int

            - num_zero: defines what we mean by numerical zero. It is used
                        to add sort attribute to both correlated and non-correlated shells.
              num_zero type : float

            - non_standard_corr_shells: may have True/False value. If True value set then no standard treatment of
                                        correlated shells. It may happen that orbitals which are usually considered
                                        non-correlated states should  be treated as correlated states,
                                        like for example O 2p  states in RbO2 can be treated as correlated states
                                        (see .Phys. Rev. B 80, 140411 (2009) for more details).
                                        If this parameter is set to True than number of shells is equal to the number
                                        of correlated shells.  This parameter should be set by a user to True only in
                                        such a non-standard  situation.

                                        If this parameter is not set (it has its default value False), then shells and
                                        correlated shells (corr_shells) are constructed according to the
                                        standard convention based on the initial projections from win file:
                                                s, p states are non-correlated states
                                                d, f states are correlated states

               non_standard_corr_shells type : bool


        """
        if is_master_node():
            reader = AsciiIO()

            extra_par_found = False
            try:
                open(self._extra_par_file, "r")
                extra_par_found = True
                self.make_statement("File with extra parameters found. "
                                    "User-defined values of verbosity, num_zero and non_standard_corr_shells keywords"
                                    "  will be used if found.")

            except IOError:
                self.make_statement("No file with extra parameters found. "
                                    "Default values of verbosity and num_zero will be used.")


            if extra_par_found:
                reader.read_ASCII_file(file_to_read=self._extra_par_file,
                                       default_dic=self._extra_par,
                                       variable_list_string_val=[])

            # set all extra parameters
            try:
                self._num_zero = float(self._extra_par["num_zero"])
            except ValueError:
                self.report_error("Wrong type of parameter num_zero!")

            try:
                self._verbosity = int(self._extra_par["verbosity"])
            except ValueError:
                self.report_error("Wrong type of parameter verbosity!")

            # check if extra parameters have reasonable values
            if not 1.0 > self._num_zero > 0.0:
                self.report_error("Numerical zero is wrongly defined!")

            if self._verbosity != 2:
                self.make_statement("No additional output will be printed out from Wannier90Converter module.")
                self._verbosity = "None"

            if isinstance(self._extra_par["non_standard_corr_shells"], bool):
                self._non_standard_corr_shells = self._extra_par["non_standard_corr_shells"]
            else:
                self.report_error("Invalid value of non_standard_corr_shells: true/false value was expected!")

        # bcasting
        self._num_zero = bcast(self._num_zero)
        self._verbosity = bcast(self._verbosity)
        self._non_standard_corr_shells = bcast(self._non_standard_corr_shells)


    def _read_win_file(self):
        """

        The following keywords are read from win file:

                *   begin projections
                    -----------------
                    -----------------
                    end projections

                *   mp_grid

                *   hr_plot (has to be set to true)

                *   num_wann

                *   unit_cell_cart

                *   num_bands (if it not present then no entanglement is assumed)

                *   fermi_energy

                *   spinors  (should be present only if  spin-orbit coupling calculations are performed)

            For the explanation of each of the keywords please look at Wannier90 user guide.


        Extracts parameters from win file(s). Parameters extracted from data stored in filename.win file;

            - chemical_potential: this parameter is equal to value of fermi_energy keyword.
              chemical_potential type: float

            - n_k: this parameter is calculated from value of mp_grid keyword.
              n_k type: int

            - shells: parameter shells is calculated form the following block:

                               Begin Projections
                                ---------------
                                ---------------
                                ---------------
                               End Projections

                        in *.win file(s). At this stage for each shell attributes "atom", "dim", "l"
                        are determined (sort is calculated in _h_to_triqs method).
              shells type : list of dictionaries which values are of type int

            - SO:   defines whether or not we have spin-orbit calculations.  If filename.win is present and
                    inside of it there is keyword  spinors=true then SO=1, and  SO=0 otherwise.
              SO type: int

            - SP:   defines whether we have spin-polarised or spin-less calculation. \n
                    Possible values:
                       If filename.win is present then SP=0,
                       if filename_up.win and filename_down.win are both present then SP=1.
              SP type: int

            While reading keywords from filename.win Fortran convention is honoured
            (cases of  letters do not matter, symbol '!' starts a comment). In addition block
            which starts from # is also considered a comment.

        """

        one_channel = False
        if is_master_node():
            reader = AsciiIO()

            n_spin = 2
            self._disentanglement_spin = {i: False for i in range(n_spin)}
            self._total_Bloch_spin = {i: 0 for i in range(n_spin)}


            if isfile(self._filename + ".win"):
                one_channel = True
            elif not (isfile(self._filename + "_up.win") and isfile(self._filename + "_down.win")):
                self.report_error("Couldn't find any of the following obligatory files (option 1) or 2) ): \n" + 
                                  "1) " + self._filename + ".win or\n" + 
                                  "2) " + self._filename + "_up.win and " + self._filename + "_down.win ")

            if one_channel:

                reader.read_ASCII_fortran_file(file_to_read=self._filename + ".win", default_dic=self._win_par)
                self._check_hr_plot(filename=self._filename + ".win", dictionary=self._win_par)
                self._get_spinors()
                self._get_total_MLWF()
                self.total_Bloch = self._get_num_bands(dictionary=self._win_par)
                self._disentanglement_spin[0] = (self.total_MLWF != self.total_Bloch)
                self._total_Bloch_spin[0] = self.total_Bloch

            else:
                temp_par = deepcopy(self._win_par)

                reader.read_ASCII_fortran_file(file_to_read=self._filename + "_up.win", default_dic=self._win_par)
                reader.read_ASCII_fortran_file(file_to_read=self._filename + "_down.win", default_dic=temp_par)

                self._check_hr_plot(filename=self._filename + "_up.win", dictionary=self._win_par)
                self._check_hr_plot(filename=self._filename + "_down.win", dictionary=temp_par)


                # Checks consistency of win files.
                for entry in self._win_par:
                    # We allow num_bands to be different for each spin channel but the rest must be the same.
                    if entry != "num_bands" and cmp(self._win_par[entry], temp_par[entry]) != 0:
                        self.report_error(self._filename + "_up.win and " + self._filename + 
                                          "_down.win  files are inconsistent!"
                                          " Entry '" + entry + "' has value '" + self._win_par[entry] + 
                                          "' in %s_up.win file " % self._filename +
                                          " and value '" + temp_par[entry] + "' in %s_down.win file." % self._filename)

                self._get_total_MLWF()
                BlochUp = self._get_num_bands(dictionary=self._win_par)
                BlochDown = self._get_num_bands(dictionary=temp_par)

                # it is assumed that for both spin channels the same number of MLWFs is constructed
                # for each spin channel disentanglement is monitored independently
                self._total_Bloch_spin[0] = BlochUp
                self._total_Bloch_spin[1] = BlochDown

                self._disentanglement_spin[0] = (self.total_MLWF != BlochUp)
                self._disentanglement_spin[1] = (self.total_MLWF != BlochDown)

                self.total_Bloch = max(BlochUp, BlochDown)

                self.SO = 0
                self.SP = 1

            # **** Calculates parameters for the converter based on the parameters from win file.
            self._get_shells()
            self._get_ham_nkpt()
            self._get_chemical_potential()
            self._get_cell_cart()

            self._n_spin = self.SP + 1 - self.SO


            self.n_shells = len(self.shells)
            self.n_k = self.ham_nkpt[0] * self.ham_nkpt[1] * self.ham_nkpt[2]  # number of k-points


            if one_channel:
                self.make_statement("The information from  %s.win file has been read." % self._filename)
            else:
                self.make_statement("The information "
                                    "from %s_up.win and %s_down.win "
                                    "files has been read." % (self._filename, self._filename))

        # bcasting
        self.shells = bcast(self.shells)
        self.n_shells = bcast(self.n_shells)
        self.ham_nkpt = bcast(self.ham_nkpt)
        self.n_k = bcast(self.n_k)
        self.SO = bcast(self.SO)
        self.SP = bcast(self.SP)
        self._n_spin = bcast(self._n_spin)
        self._disentanglement_spin = bcast(self._disentanglement_spin)
        self.total_Bloch = bcast(self.total_Bloch)
        self._total_Bloch_spin = bcast(self._total_Bloch_spin)
        self.total_MLWF = bcast(self.total_MLWF)
        self.chemical_potential = bcast(self.chemical_potential)
        self._wannier_order = bcast(self._wannier_order)
        self._name2SpinChannel = bcast(self._name2SpinChannel)
        self._win_par = bcast(self._win_par)


    def _get_spinors(self):
        """
        Calculates spinors from data which was extracted from *.win file. Sets value of SO.

        """

        if not (self._win_par["spinors"].lower() == "true" or
                self._win_par["spinors"].lower() == "false" or
                self._win_par["spinors"].lower() == "t" or
                self._win_par["spinors"].lower() == "f"):
            self.report_error("Invalid value of spinors!")

        spinors = (self._win_par["spinors"].lower() == "true" or self._win_par["spinors"].lower() == "t")

        if spinors:
            self.SO = 1
            self.SP = 1
        else:
            self.SO = 0
            self.SP = 0


    def _get_chemical_potential(self):

        """
        Calculates initial guess for chemical potential from data which was extracted from *.win file.

        """

        try:
            self.chemical_potential = float(self._win_par["fermi_energy"])
        except ValueError:
            self.report_error("Invalid value of fermi_energy! ")


    def _get_cell_cart(self):
        """

        Calculates lattice vectors.
        """
        dim = 3
        lattice = self._win_par["unit_cell_cart"].split("\n")
        unit = None
        bohr2ang = 0.52917721092  # one Bohr in Ang (from wannier90-2.0.1: constants.f90, L88)
        self._cell_cart = numpy.zeros((dim, dim), dtype=numpy.float_)

        if "bohr" in lattice[0].lower() or "ang" in lattice[0].lower():
            unit = lattice.pop(0).lower()

        try:
            for i, line in enumerate(lattice):
                for j, item in enumerate(line.split()):
                    self._cell_cart[i, j] = float(item)
        except ValueError:
            self.report_error("Invalid cell lattice vectors!")

        if unit == "bohr":
            self._cell_cart *= bohr2ang


    def _get_ham_nkpt(self):
        """
        Calculates vectors with 3 elements which defines k-mesh
        from data which was extracted from *.win file.

        """

        dim = 3
        temp_mp_grid = self._win_par["mp_grid"].split()
        self.ham_nkpt = []


        try:
            for i in range(len(temp_mp_grid)): self.ham_nkpt.append(int(temp_mp_grid[i]))
        except ValueError:
            self.report_error("Invalid value of mp_grid!")

        if not (len(self.ham_nkpt) == dim and all([self.ham_nkpt[i] > 0 for i in range(len(self.ham_nkpt))])):
            self.report_error("You have to define a proper k-mesh!")


    def _get_shells(self):
        """

        Calculates shells  basing on
        data which was extracted from *.win file.
        Value of "sort" keyword  is found later from H_R by _h_to_triqs method.

        """
        lines = self._win_par["projections"].split("\n")

        self.shells = []
        self._atoms = {}
        self._wannier_order = []
        num_atom = -1
        num_shell = -1

        if "random" in str(lines).lower():

                self.report_error("Random projections detected. Random projections are not "
                                  "valid for the Wannier90Converter. Initial projections should be as close"
                                  " as possible to the final MLWFs.")

        if ("(u" in str(lines) or "(d" in str(lines)) and ")" in str(lines):

                self.report_error("Seems that you try to define spinors in the format like: \n"
                                  "Fe:dxy;dxz;dyz(u)\n"
                                  " or \n"
                                  "Fe:dxy;dxz;dyz(d) \n"
                                  " or \n"
                                  "Fe:dxy;dxz;dyz(d,u). \n"
                                  "This format is not supported by the converter. "
                                  "The number of spinors for 'up channel' "
                                  "is expected to be equal to the number of spinors for the 'down chanel'."
                                  "The following format is accepted:\n"
                                  "Fe:dxy;dxz;dyz")

        if ":" not in lines[0]: lines.pop(0)  # in case first line in the block is [units] remove it

        for line in lines:  # each line is a different atom, but atoms can repeat themselves


            label = (line[:line.find(":")])
            begin = line.find(":") + 1  # here we add 1 because we want content after

            if ":" in line[begin:]:
                end = line[begin:].find(":") + begin
            else:
                end = len(line)

            projections = line[begin:end].split(";")

            if label not in self._atoms:

                num_atom += 1
                num_shell += 1
                self._atoms[label] = {"num_atom": num_atom, "state": []}
                res = self._get_proj_l(input_proj=projections.pop(0), atom_name=label)
                temp_shells = [{"atom": num_atom, "dim": res["dim"], "l": res["l"]}]

                wannier_order = [res["name"]]
                for proj in projections:
                    res = self._get_proj_l(input_proj=proj, atom_name=label)
                    l = res["l"]
                    dim = res["dim"]
                    found = False
                    for i, item in enumerate(temp_shells):
                        if l == item["l"]:
                            item["dim"] += dim
                            wannier_order[i].extend(res["name"])
                            found = True
                            break
                    if not found:  # different shell on the same atom, with different l
                        temp_shells.append({"atom": num_atom, "dim": dim, "l": l})
                        wannier_order.append(res["name"])

                self.shells.extend(temp_shells)
                self._wannier_order.extend(wannier_order)

            else:

                local_num_atom = self._atoms[label]["num_atom"]
                for proj in projections:
                    res = self._get_proj_l(input_proj=proj, atom_name=label)
                    l = res["l"]
                    dim = res["dim"]
                    found = False
                    for i, item in enumerate(self.shells):
                        if l == item["l"] and item["atom"] == local_num_atom:
                            item["dim"] += dim
                            self._wannier_order[i].extend(res["name"])
                            found = True
                            break
                    if not found:  # different shell on the same atom, with different l
                        self.shells.extend({"atom": local_num_atom, "dim": dim, "l": l})
                        self._wannier_order.extend(res["name"])


        # check if shells are reasonable
        for state in self.shells:

            if state["l"] < 0:  # non-correlated hybridized states
                hyb_found = False
                for entry in self._all_states:
                    if state["l"] == entry["l"] and state["dim"] != entry["dim"]:
                        hyb_found = True
                        break
                if not hyb_found:
                    self.report_error("Wrong structure of shells!")
            elif state["dim"] > (state["l"] * 2 + 1):
                self.report_error("Wrong structure of shells!")


    def _get_corr_shells(self):

        """

        Calculates corr_shells from shells.

        """
        self._wannier_order_corr = []

        if self.shells is None: self.report_error("Shells not set yet. They are needed to calculate corr_shells!")

        if self._non_standard_corr_shells:
            self.corr_shells = deepcopy(self.shells)
            self._wannier_order_corr = self._wannier_order
            for n_corr in range(self.n_shells):
                self.corr_shells[n_corr]["SO"] = self.SO
                self.corr_shells[n_corr]["irep"] = 0
        else:
            self.corr_shells = []
            for i, state in enumerate(self.shells):
                if state["l"] == 2 or state["l"] == 3:
                    corr_state = deepcopy(state)
                    corr_state["SO"] = self.SO
                    corr_state["irep"] = 0
                    self.corr_shells.append(corr_state)
                    self._wannier_order_corr.append(self._wannier_order[i])

        self.n_corr_shells = len(self.corr_shells)
        
        if self.n_corr_shells == 0:
            self.report_error("No correlated shells detected in the system!")


    def _get_proj_l(self, input_proj=None, atom_name=None):
        """
        Returns attributes of the particular initial projection.
        :param input_proj: particular initial projection (initial guess for MLWFs) to be analysed.
        :type input_proj: str

        :param atom_name: Name of atom from *.win file for which shells are to be found
        :type atom_name: str

        :return: l ,dim, name for  the given initial guess for MLWFs
        :rtype : dict (dictionary)
        """

        proj_all_entries = input_proj.replace(",", " ").replace("=", " ").lower().split()
        l = None
        dim = 0
        indx = 0
        name = []

        if not isinstance(input_proj, str):
            self.report_error("Projection %s is an invalid initial projection!" % proj_all_entries)

        # case: l=foo1, mr=foo2,foo3,...
        if len(proj_all_entries) >= 4:

            if "l" in proj_all_entries:
                indx = proj_all_entries.index("l") + 1
                if indx == len(proj_all_entries): self.report_error("Projection %s is an "
                                                                    "invalid initial projection!" % proj_all_entries)

                # converts value l from string to int
                try:
                    l = int(proj_all_entries[indx])
                except ValueError:
                    self.report_error("Invalid value of l!")

            else:
                self.report_error("Projection %s is an invalid"
                                  " initial projection!" % proj_all_entries)

            indx += 1
            if not "mr" == proj_all_entries[indx]:

                self.report_error("Invalid initial projection! No  mr keyword found!")

            mr_list = proj_all_entries[indx + 1:]  # stores all mr values

            # converts string elements to int values
            try:
                for mr_indx in range(len(mr_list)):
                    mr_list[mr_indx] = int(mr_list[mr_indx])
            except ValueError:
                self.report_error("Invalid value of mr!")

            for mr_item in mr_list:

                mr_found = False

                for state in self._all_states:

                    if state["l"] == l and state["mr"] == mr_item:
                        dim += state["dim"]
                        mr_found = True
                        name.append(state["name"])

                        # checks if the same projection not repeated
                        self._is_unique_proj(atom_name=atom_name, state=state)

                        break

                if not mr_found:
                    self.report_error("Invalid value of mr!")

        # case: l=foo
        elif len(proj_all_entries) == 2:
            if "l" in proj_all_entries:
                indx = proj_all_entries.index("l") + 1
                if indx == len(proj_all_entries): self.report_error("Projection %s is an invalid"
                                                                    " initial projection!" % proj_all_entries)
                l = proj_all_entries[indx]
            else:
                self.report_error("Projection %s is an invalid "
                                  "initial projection!" % proj_all_entries)

            for state in self._all_states:

                if state["l"] == l and state["mr"] is None:
                    dim = state["dim"]
                    name.append(state["name"])

                    # checks if the same projection not repeated
                    self._is_unique_proj(atom_name=atom_name, state=state)

                    break

            if dim == 0:
                self.report_error("Invalid value of l!")

        # case: foo
        elif len(proj_all_entries) == 1:

            for state in self._all_states:
                if state["name"] == proj_all_entries[0]:
                    l = state["l"]
                    dim = state["dim"]
                    name.append(state["name"])
                    # checks if the same projection not repeated
                    self._is_unique_proj(atom_name=atom_name, state=state)
                    break

            if dim == 0:
                self.report_error("Projection %s is an invalid "
                                  "initial projection!" % proj_all_entries)

        else:
            self.report_error("Projection %s is an invalid "
                              "initial projection!" % proj_all_entries)


        return {"l": l, "dim": dim, "name": name}


    def _get_total_MLWF(self):
        """
        Converts number of MLWFs in the form of string from *.win file to Wannier90Converter attribute: total_MLWF.
        """

        try:
            self.total_MLWF = int(self._win_par["num_wann"])
        except ValueError:
            self.report_error("Invalid number of MLWFs!")


    def _check_hr_plot(self, filename=None, dictionary=None):
        """
        Checks if there is a valid hr_plot keyword in filename.

        :param filename: name of the win file in which search for a keyword 'hr_plot' was made
        :type filename: str
        :param dictionary: dictionary which stores data (values are strings) extracted from win file
        :type dictionary: dict

        """

        if not (dictionary["hr_plot"].lower() == "true" or
                dictionary["hr_plot"].lower() == "t"):
            self.report_error("No valid 'hr_plot' keyword was found in %s!" % filename)


    def _get_num_bands(self, dictionary=None):

        """
        Converts number of Bloch states in the form of string from *.win
        file to the number of Bloch states represented as an integer number.

        :param dictionary: dictionary with "num_bands" entry.
        :type dictionary: dict
        """
        try:
            if dictionary["num_bands"] == -1:
                return self.total_MLWF
            else:
                return int(dictionary["num_bands"])
        except ValueError:
            self.report_error("no valid num_bands keyword in the dictionary!")


    def _get_T(self):
        """
        Determines T for all inequivalent correlated shells.
        """

        self.T = []
        for n_corr in range(self.n_inequiv_corr_shells):

            self.T.append(self._T_order_orb(n_corr=n_corr))


    def _T_order_orb(self, n_corr=None):
        """
        Determines T for the particular inequivalent correlated shell.

        :param n_corr: Number of inequivalent correlated shell for
                       which the transformation matrix should be constructed.
        :type n_corr: int

        :return:  transformation  matrix T
        :rtype : numpy.array((2 * l + 1) * (1 + SO), (2 * l + 1) * (1 + SO)),dtype=numpy.complex_)
        """

        SO = self.corr_shells[self.inequiv_to_corr[n_corr]]["SO"]

        if SO == 1:

            # This part is based on  lines  L360-L468 from outputqmc.f (dmftproj subpackage in main TRIQS)
            # where T matrix in the format accepted by  Wien2kConverter is evaluated.

            # Only the case of a basis without "mixing" is implemented.

            l = self.corr_shells[self.inequiv_to_corr[n_corr]]["l"]
            half_dim = (l * 2 + 1)
            dim = 2 * half_dim
            T = numpy.zeros((dim, dim), dtype=numpy.complex_)

            T_block = self._get_order(n_corr=n_corr)

            T[0:half_dim, 0:half_dim] = T_block
            T[half_dim:dim, half_dim:dim] = T_block

        else:

            T = self._get_order(n_corr=n_corr)

        return T


    def _get_order(self, n_corr=None):
        """
        In case of spin-orbit coupling determines block of T for the
        inequivalent correlated shell, in other cases returns the whole T for the
        inequivalent correlated shell.

        :param n_corr: Number of inequivalent correlated shell for
                       which the transformation matrix should be constructed.
        :type n_corr: int

        :return:  transformation  matrix T
        :rtype : numpy.array(2 * l + 1, 2 * l + 1 ),dtype=numpy.complex_)
        """


        a = 1.0 / math.sqrt(2)
        b = 1.0j / math.sqrt(2)

        l = self.corr_shells[self.inequiv_to_corr[n_corr]]["l"]
        dim = l * 2 + 1

        if l == 0:

            Order     = {"s": numpy.array([1.0])}

        elif l == 1:

            Order     = {
                            "px": numpy.array([a  ,  0.0, -a  ]),
                            "py": numpy.array([b  ,  0.0,  b  ]),
                            "pz": numpy.array([0.0,  1.0,  0.0])
                        }

        elif l == 2:

            Order     = {
                            "dz2":     numpy.array([ 0.0,  0.0,  1.0,  0,  0.0]),
                            "dx2-y2":  numpy.array([ a  ,  0.0,  0.0,  0,  a  ]),
                            "dxy":     numpy.array([-b  ,  0.0,  0.0,  0,  b  ]),
                            "dyz":     numpy.array([ 0.0,  b  ,  0.0, -b,  0.0]),
                            "dxz":     numpy.array([ 0.0,  a  ,  0.0,  a,  0.0])
                        }

        elif l == 3:

            Order     = {
                            "fx(x2-3y2)":   numpy.array([a  ,  0.0,  0.0,  0.0,  0.0,  0.0,  -a ]),
                            "fz(x2-y2)":    numpy.array([0.0,  a  ,  0.0,  0.0,  0.0,  a,    0.0]),
                            "fxz2":         numpy.array([0.0,  0.0,  a,    0.0, -a  ,  0.0,  0.0]),
                            "fz3":          numpy.array([0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0]),
                            "fyz2":         numpy.array([0.0,  0.0,  b,    0.0,  b  ,  0.0,  0.0]),
                            "fxyz":         numpy.array([0.0,  b  ,  0.0,  0.0,  0.0,  -b,   0.0]),
                            "fy(3x2-y2)":   numpy.array([b  ,  0.0,  0.0,  0.0,  0.0,  0.0,    b])
                        }

        else:

            self.report_error("Unsupported value of l parameter: %s!" % l)


        names_map = { "s":         ["s"],
                      "p":         ["pz", "px", "py"],  # the order taken from Wannier90 user guide (p. 43)
                      "pz":        ["pz"],
                      "px":        ["px"],
                      "py":        ["py"],
                      "d":         ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],  # the order taken from Wannier90 user guide (p. 43)
                      "dz2":       ["dz2"],
                      "dxz":       ["dxz"],
                      "dyz":       ["dyz"],
                      "dx2-y2":    ["dx2-y2"],
                      "dxy":       ["dxy"],
                      "f":         ["fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"],  # the order taken from Wannier90 user guide (p. 43)
                      "fz3":       ["fz3"],
                      "fxz2":      ["fxz2"],
                      "fyz2":      ["fyz2"],
                      "fz(x2-y2)": ["fz(x2-y2)"],
                      "fxyz":      ["fxyz"],
                      "fx(x2-3y2)":["fx(x2-3y2)"],
                      "fy(3x2-y2)":["fy(3x2-y2)"]
                    }

        order_wannier = []

        for item in self._wannier_order_corr[self.inequiv_to_corr[n_corr]]:

                order_wannier.extend(names_map[item])

        if len(order_wannier) == dim:

            T = numpy.zeros((dim, dim), dtype=numpy.complex_)
            for i, name in enumerate(order_wannier):

                T[i, :] = Order[name][:]

        else:

            # in case number of orbitals is different than max number of orbitals generic T matrix is constructed,
            # and it does not depend on the order of the initial projections from the win file.

            T = spherical_to_cubic(l=l)

        return T


    def _get_density_required(self):
        """
        Calculates a required occupancy of the local Hamiltonian basing on the Fermi
        energy from *.win file and H_k.
        """

        occ = 0.0
        indices_kpt = numpy.array(range(self.n_k))
        for n_s in range(self._n_spin):
            occ_part = 0.0
            # parallelization over n_k
            for ikpt in slice_array(indices_kpt):
                temp_eig, temp_rot_mat = numpy.linalg.eigh(self.H_k[ikpt][n_s])
                for eig in temp_eig:
                    if eig <= self.chemical_potential:
                        occ_part += 1.0
            occ += all_reduce(world, occ_part, lambda x, y: x + y)
            barrier()
        self.density_required = occ / self.n_k
        # case of the spinless calculation
        if self._n_spin == 1 and self.SO == 0:
            self.density_required *= 2


    def _h_to_triqs(self):
        """
        Reads  H_R file calculated by Wannier90  and calculates H_k and symmetry operations.
        Detailed steps below:

            * Reads filename_hr.dat

            * Evaluates sort parameter for shells

            * Finds correlated shells

            * Constructs symmetry operators

            * Constructs H_k

            * Updates parameters dictionary

        """

        LOCAL_VARIABLES = ["ham_r", "nrpt"]

        GLOBAL_VARIABLES = ["FULL_H_R",
                            "Vector_R_degeneracy",
                            "Vector_R"]

        for it in LOCAL_VARIABLES:
            exec "% s = 0" % it

        for it in GLOBAL_VARIABLES:
            exec "self.%s=0" % it

        if self._n_spin == 2:
            results = self.__read_spin_H_R

        else:
            results = self._read_H_R_file(filename=self._filename)

        for it in GLOBAL_VARIABLES:
            exec "self.%s = results[it]" % it

        for it in LOCAL_VARIABLES:
            exec "% s =  results[it]" % it


        # in case of spin-orbit coupling we have to rearrange  "ham_r" and "FULL_H_R":
        # 1_up 1_down 2_up 2_down ..... n_up n_down   to
        # 1_up 2_up ....n_up 1_down 2_down ..... n_down

        if self.SO == 1:

            temp_full_H_R = deepcopy(self.FULL_H_R)
            temp_ham_r = deepcopy(ham_r)

            num_wann_channel = self.total_MLWF / 2
            ind = range(0, self.total_MLWF, 2)

            for ii, ind_i in enumerate(ind):
                for jj, ind_j in enumerate(ind):
                    for irpt in range(self.n_k):
                        self.FULL_H_R[irpt][ii, jj] = temp_full_H_R[irpt][ind_i, ind_j]
                        self.FULL_H_R[irpt][ii + num_wann_channel, jj + num_wann_channel] = temp_full_H_R[irpt][ind_i + 1, ind_j + 1]
                    ham_r[ii, jj] = temp_ham_r[ind_i, ind_j]
                    ham_r[ii + num_wann_channel, jj + num_wann_channel] = temp_ham_r[ind_i + 1, ind_j + 1]
                    
                    
            del temp_full_H_R
            del temp_ham_r


        if results["total_MLWF"] != self.total_MLWF:
            self.report_error("Inconsistent number of MLWFs in %s_hr.dat"
                              " and in %s.win files" % (self._filename, self._filename))

        # 2) Evaluates "sort" for shells. Evaluation is based on the comparison of
        #    eigenvalues of H_00 block for each shell. H_00 is a real representation of
        #    the Hamiltonian represented in MLWFs which corresponds to R=[0 ,0 ,0 ]

        total_orb = 0
        last_sort = -1
        all_sorts = {}

        for ish in range(self.n_shells):
            old_sort_found = False
            n_orb = self.shells[ish]["dim"]
            temp_eig = numpy.linalg.eigvalsh(ham_r[total_orb: total_orb + n_orb, total_orb:total_orb + n_orb])
            t1 = temp_eig.argsort()
            for sort_item in all_sorts:
                 if cmp(temp_eig.shape, all_sorts[sort_item].shape) == 0 and self._num_zero > numpy.min(numpy.abs(temp_eig[t1] - all_sorts[sort_item])):
                     old_sort_found = True
                     self.shells[ish]["sort"] = sort_item
                     break

            if not old_sort_found:
                 last_sort += 1
                 self.shells[ish]["sort"] = last_sort
                 all_sorts[last_sort] = deepcopy(temp_eig[t1])  # save sorted values

            total_orb += n_orb

        # 3)  Finds correlated shells
        self._get_corr_shells()


        # Determines the number of inequivalent correlated shells (taken from ConverterTools)
        self.n_inequiv_corr_shells, self.corr_to_inequiv, self.inequiv_to_corr = self.det_shell_equivalence(corr_shells=self.corr_shells)


        # 4) Constructs symmetry operators
        self._R_sym = []
        for n_corr in range(self.n_corr_shells):

            self._R_sym.append({"atom": self.corr_shells[n_corr]["atom"],
                                "sort": self.corr_shells[n_corr]["sort"],
                                "rot_mat": None,
                                "eig": None})

            # R_sym is  data structure to store symmetry operation, it has a form of list,
            # each entry correspond to one correlated shell, entry is a dictionary with following keywords:
            #   "atom":     number of atom
            #   "sort" :    label of symmetry equivalent atom, all equivalent atoms have the
            #               same sort
            #   "rot_mat":  rotation matrix numpy.zeros((self.corr_shells[n_corr]['dim'],
            #                                            self.corr_shells[n_corr]['dim']),
            #                                            numpy.complex)
            #   "eig":      eigenvalues which correspond to the particular correlated site
            #               numpy.zeros(self.corr_shells[n_corr]['dim'],dtype=numpy.float)


        local_shells = [{} for n in range(self.n_shells)]
        total_orb = 0

        for ish in range(self.n_shells):

            n_orb = self.shells[ish]["dim"]
            temp_eig, temp_rot_mat = numpy.linalg.eigh(ham_r[total_orb: total_orb + n_orb, total_orb: total_orb + n_orb])
            # sort eigenvectors and eigenvalues
            temp_indx = temp_eig.argsort()
            local_shells[ish]["eig"] = temp_eig[temp_indx]
            local_shells[ish]["rot_mat"] = temp_rot_mat[:, temp_indx]

            # Checks if shell is a correlated shell if so write eigenvectors to rotation which corresponds to it
            for icrsh in range(self.n_corr_shells):

                if self.compare_shells(shell=self.shells[ish],
                                       corr_shell=self.corr_shells[icrsh]):

                    self._R_sym[icrsh]["eig"] = local_shells[ish]["eig"]
                    self._R_sym[icrsh]["rot_mat"] = local_shells[ish]["rot_mat"]
                    break

            # in case of spin-polarised calculation we run only through "up"
            # block of ham_r, it is ok because we look for a spacial symmetry here
            total_orb += n_orb

        del local_shells

        if self._win_par["spinors"].lower() == "true" or self._win_par["spinors"].lower() == "t":
            # in case of spin-orbit  num_wann = 2 * trial projections are provided
            if self.total_MLWF != sum([sh['dim'] for sh in self.shells]) * 2:

                self.report_error("Wrong block structure of input Hamiltonian, correct it please!")
        else:
            if self.total_MLWF != sum([sh['dim'] for sh in self.shells]):

                self.report_error("Wrong block structure of input Hamiltonian, correct it please!")

        self.rot_mat = [numpy.zeros((self.corr_shells[icrsh]['dim'], self.corr_shells[icrsh]['dim']),
                                    numpy.complex_) for icrsh in range(self.n_corr_shells)]

        for icrsh in range(self.n_corr_shells):
            try:
                self.rot_mat[icrsh] = numpy.dot(self._R_sym[icrsh]["rot_mat"],
                                                self._R_sym[self.corr_to_inequiv[icrsh]]["rot_mat"].conjugate().transpose())
            except ValueError:
                self.report_error("Rotation matrices cannot be constructed. Are valid user-provided shells in use?")

        # 5) Constructs H_k
        self.H_k = [[numpy.zeros((self.total_MLWF, self.total_MLWF), dtype=numpy.complex_)
                    for isp in range(self._n_spin)] for ikpt in range(self.n_k)]

        self.__make_k_point_mesh()
        imag = 1j
        twopi = 2 * numpy.pi

        total_n_kpt = self.ham_nkpt[0] * self.ham_nkpt[1] * self.ham_nkpt[2]

        for ikpt in range(total_n_kpt):
            for n_s in range(self._n_spin):
                indices_nrpt = numpy.array(range(nrpt))
                # parallelization over nrpt
                for irpt in slice_array(indices_nrpt):
                    rdotk = twopi * numpy.dot(self.k_point_mesh[ikpt], self.Vector_R[irpt])
                    factor = (math.cos(rdotk) + imag * math.sin(rdotk)) / float(self.Vector_R_degeneracy[irpt])

                    self.H_k[ikpt][n_s][:, :] += factor * self.FULL_H_R[irpt][self.total_MLWF * n_s:
                                                                             self.total_MLWF * (n_s + 1),
                                                                             self.total_MLWF * n_s:
                                                                             self.total_MLWF * (n_s + 1)]

                self.H_k[ikpt][n_s][:, :] = all_reduce(world, self.H_k[ikpt][n_s][:, :], lambda x_ham, y: x_ham + y)

                barrier()

        # 6) Updates parameters dictionary

        Variables = {"k_point_mesh": self.k_point_mesh, "FULL_H_R": self.FULL_H_R,
                     "Vector_R": self.Vector_R,
                     "Vector_R_degeneracy": self.Vector_R_degeneracy,
                     "Hamiltonian": self.H_k,
                     "R_sym": self._R_sym}

        self._parameters.update(Variables)


        if self._n_spin == 2:
            self.make_statement(" % s_up_hr.dat and %s_down_hr.dat"
                                " files have been read." % (self._filename, self._filename))
        else:
            self.make_statement(" % s_hr.dat file has been read." % self._filename)


    def _read_H_R_file(self, filename=None):

        """
        Reads H_R from the ASCII file. The same format as for seedname_hr.dat output Wannier90 file is expected.
        :param filename: name of H_R file to read
        :type filename: str
        :return: dictionary with data which was read form filename_hr.dat

        """

        local_variable = {"nrpt": None, "total_MLWF": None,
                          "Vector_R_degeneracy": None,
                          "Vector_R": None, "FULL_H_R": None, "ham_r": None}

        if is_master_node():
            try:
                with open(filename + "_hr.dat", "rb") as hr_txt_file:

                    hr_file = hr_txt_file.readlines()
                    hr_txt_file.close()
                    dimensions = 3
                    pos = 0  # first line == date

                    pos += 1  # total_MLWFs
                    try:
                        total_MLWF = int(hr_file[pos])  # reads number of Wannier functions per spin
                    except ValueError:
                        self.report_error("Invalid number of MLWFs in %s_hr.dat" % filename)

                    pos += 1  # nrpt
                    try:
                        nrpt = int(hr_file[pos])
                    except ValueError:
                        self.report_error("Invalid number of R vectors!")

                    Vector_R_degeneracy = numpy.zeros(nrpt, dtype=int)
                    Vector_R = numpy.zeros((nrpt, dimensions), dtype=int)

                    FULL_H_R = [numpy.zeros((total_MLWF, total_MLWF), dtype=numpy.complex_) for n in range(nrpt)]

                    ham_r = numpy.zeros((total_MLWF, total_MLWF), dtype=numpy.complex_)

                    counter = 0
                    while counter < nrpt:
                        pos += 1
                        for x in hr_file[pos].split():
                            try:
                                 Vector_R_degeneracy[counter] = int(x)
                            except ValueError:
                                 self.report_error("Cannot read degeneracy of R vectors from %s_hr.dat!" % filename)
                            counter += 1

                    for irpt in range(nrpt):
                        for jj in range(total_MLWF):
                            for ii in range(total_MLWF):
                                pos += 1

                                try:
                                    line = [float(x) for x in hr_file[pos].split()]
                                except ValueError:
                                    self.report_error("Can't read Hamiltonian in the"
                                                      " real representation from %s_hr.dat" % filename)

                                if ii == 0 and jj == 0:
                                    Vector_R[irpt] = numpy.array([int(line[0]), int(line[1]), int(line[2])])

                                indx_i = int(line[3]) - 1  # we count from zero
                                indx_j = int(line[4]) - 1
                                if ii != indx_i or jj != indx_j:
                                   self.report_error("Inconsistent indices "
                                                     "for H_R: [%s!=%s,%s!=%s]!" % (ii, indx_i, jj, indx_j))

                                FULL_H_R[irpt][ii, jj] = complex(float(line[5]), float(line[6]))
                                if int(line[0]) == 0 and int(line[1]) == 0 and int(line[2]) == 0:
                                    ham_r[ii, jj] = complex(float(line[5]), float(line[6]))  # zeroth unit cell found

                        # checks if  FULL_H_R[irpt] is real, for well localized MLWFs H_R should be real
                        if self._num_zero < numpy.abs((FULL_H_R[irpt].imag.max()).max()):
                             self.report_warning("H_R in MLWF has large complex components!")

                    # checks if ham_r is Hermitian
                    if not numpy.allclose(ham_r.transpose().conjugate(), ham_r, atol=self._num_zero):
                        self.report_warning("Your Hamiltonian is not symmetric!")

                local_variable.update({"nrpt": nrpt, "total_MLWF": total_MLWF,
                                       "Vector_R_degeneracy": Vector_R_degeneracy,
                                       "Vector_R": Vector_R, "FULL_H_R": FULL_H_R,
                                       "ham_r": ham_r})

            except IOError:
                self.report_error("Opening file %s_hr.dat failed!" % filename)

        local_variable = bcast(local_variable)
        barrier()

        return local_variable


    @property
    def __read_spin_H_R(self):
        """
        In case of spin-polarised calculations two files  filename+[up, down]'_hr.dat'
        are obtained from wannier90.x. Each file corresponds to the different
        spin channel. This method produces from those files a full Hamiltonian H_R (no-spin mixing).
        """

        # initialize it so that result variable is visible to all threads
        result = {"nrpt": None, "total_MLWF": None,
                  "Vector_R_degeneracy": None,
                  "Vector_R": None, "FULL_H_R": None, "ham_r": None}

        # reads spin channel up
        results = self._read_H_R_file(filename=self._filename + "_up")
        nrpt_up = results["nrpt"]
        total_MLWF_up = results["total_MLWF"]
        Vector_R_degeneracy_up = results["Vector_R_degeneracy"]
        Vector_R_up = results["Vector_R"]
        FULL_H_R_up = results["FULL_H_R"]
        ham_r_up = results["ham_r"]

        # reads spin channel down
        results = self._read_H_R_file(filename=self._filename + "_down")
        nrpt_down = results["nrpt"]
        total_MLWF_down = results["total_MLWF"]
        Vector_R_degeneracy_down = results["Vector_R_degeneracy"]
        Vector_R_down = results["Vector_R"]
        FULL_H_R_down = results["FULL_H_R"]
        ham_r_down = results["ham_r"]


        # nrpt, total_MLWF, Vector_R_degeneracy, Vector_R must be the same for each spin channel

        # FULL_H_R, ham_r are expected to be different for each spin channel

        if nrpt_up != nrpt_down:
            self.report_error(self._filename + 
                              "down_hr.dat has a different grid " +
                              "in real space  than " + 
                              self._filename + "up_hr.dat!")

        if total_MLWF_up != total_MLWF_down:
            self.report_error(self._filename + 
                              "down_hr.dat has a different number " +
                              "of Wannier functions than " + 
                              self._filename + "up_hr.dat!")

        if not (Vector_R_degeneracy_up.shape == Vector_R_degeneracy_down.shape and
                numpy.allclose(Vector_R_degeneracy_up, Vector_R_degeneracy_down)):

            self.report_error("Inconsistent degeneracies of R-vectors in channel up and down!")

        if not (Vector_R_up.shape == Vector_R_down.shape and
                numpy.allclose(Vector_R_up, Vector_R_down)):

            self.report_error("Inconsistent R vectors for channel up and down!")


        # produces a combined Hamiltonian
        combined_num_orb = self._n_spin * total_MLWF_down
        FULL_H_R = [numpy.zeros((combined_num_orb, combined_num_orb),
                                dtype=numpy.complex_)
                    for n in range(nrpt_down)]

        ham_r = numpy.zeros((combined_num_orb, combined_num_orb), dtype=numpy.complex_)

        for n in range(nrpt_down):
            FULL_H_R[n][:total_MLWF_up, :total_MLWF_up] = FULL_H_R_up[n]
            FULL_H_R[n][total_MLWF_down:combined_num_orb, total_MLWF_down:combined_num_orb] = FULL_H_R_down[n]

        ham_r[:total_MLWF_up, :total_MLWF_up] = ham_r_up
        ham_r[total_MLWF_down:combined_num_orb, total_MLWF_down:combined_num_orb] = ham_r_down

        result.update({"nrpt": nrpt_down, "total_MLWF": total_MLWF_down,
                       "Vector_R_degeneracy": Vector_R_degeneracy_down,
                       "Vector_R": Vector_R_down, "FULL_H_R": FULL_H_R, "ham_r": ham_r})

        return result


    def __make_k_point_mesh(self):
        """
        Makes uniformly distributed k-point mesh.

        """
        if self.ham_nkpt[0] % 2:
            i1min = - (self.ham_nkpt[0] - 1) / 2
            i1max = (self.ham_nkpt[0] - 1) / 2
        else:
            i1min = -(self.ham_nkpt[0] / 2 - 1)
            i1max = self.ham_nkpt[0] / 2

        if self.ham_nkpt[1] % 2:
            i2min = - (self.ham_nkpt[1] - 1) / 2
            i2max = (self.ham_nkpt[1] - 1) / 2
        else:
            i2min = -(self.ham_nkpt[1] / 2 - 1)
            i2max = self.ham_nkpt[1] / 2

        if self.ham_nkpt[2] % 2:
            i3min = - (self.ham_nkpt[2] - 1) / 2
            i3max = (self.ham_nkpt[2] - 1) / 2

        else:
            i3min = -(self.ham_nkpt[2] / 2 - 1)
            i3max = self.ham_nkpt[2] / 2
        dimensions = 3
        self.k_point_mesh = numpy.zeros((self.n_k, dimensions), dtype=numpy.float_)
        n_kpt = 0
        for i1 in range(i1min, i1max + 1):
            for i2 in range(i2min, i2max + 1):
                for i3 in range(i3min, i3max + 1):
                    self.k_point_mesh[n_kpt, :] = [float(i1) / float(self.ham_nkpt[0]),
                                                   float(i2) / float(self.ham_nkpt[1]),
                                                   float(i3) / float(self.ham_nkpt[2])]
                    n_kpt += 1


    def _is_unique_proj(self, atom_name=None, state=None):

        """
        Checks if a projection was not defined before. If it was defined before then
        prints info about repetition and terminates.

        :param state: dictionary with the projection
        :type state: dict
        """

        if state not in self._atoms[atom_name]["state"]:
            self._atoms[atom_name]["state"].append(state)
        else:
            self.report_error("Repetition of the projection for %s"
                              " for atom labeled by: %s" % (state, atom_name))


    def is_shell(self, shell_1st=None, shell_2nd=None):
        """
        Checks if shell_1st is equivalent to shell_2nd
        (it is defined to be the same when l, dim, sort are the same)

        :param shell_1st: first shell to compare
        :type shell_1st: dict of int

        :param shell_2nd:  second shell to compare
        :type shell_2nd: dict of int

        :return: True if shell_1st is equivalent to the shell_2nd, and False otherwise

         """

        if not self.check_shell(shell_1st, t=self.__class__.shells_keywords):
             self.report_error("Shell was expected!")
        if not self.check_shell(shell_2nd, t=self.__class__.shells_keywords):
             self.report_error("Shell was expected!")

        return (shell_1st["dim"]  == shell_2nd["dim"] and
                shell_1st["l"]    == shell_2nd["l"] and
                shell_1st["sort"] == shell_2nd["sort"])


    def compare_shells(self, shell=None, corr_shell=None):
        """
        Checks if shell corresponds to corr_shell

        :param shell: shell to compare
        :type shell: dict of int
        :param corr_shell:  correlated shell to compare
        :type corr_shell: dict of int
        :return: True if shell corresponds to corr_shell, and False otherwise
        :rtype: bool
        """

        if not self.check_shell(x=corr_shell, t=self.__class__.corr_shells_keywords):
            self.report_error("Correlated shell was expected!")
        if not self.check_shell(x=shell, t=self.__class__.shells_keywords):
            self.report_error("Shell was expected!")

        return (shell["atom"] == corr_shell["atom"] and
                shell["sort"] == corr_shell["sort"] and
                shell["l"] == corr_shell["l"] and
                shell["dim"] == corr_shell["dim"])


    def eval_offset(self, n_corr=None):
        """

        Calculates a position of the block in the matrix which corresponds
        to the n_corr-th correlated shell. A position is defined by
        offset: beginning of block ('upper left corner' of block)
        and size of block (its 'width').

        :param n_corr: number of correlated shell
        :type n_corr: int
        :return:  * n_orb: number of states for the n_corr-th correlated shell

                  * offset: position of block in matrix (for example U matrix )
                    which corresponds to the n_corr-th correlated shell
        :rtype: tuple of ints
        """

        if not self.check_n_corr(n_corr=n_corr):
            self.report_error("Invalid number of correlated shell!")

        # calculates the offset:
        offset = 0
        n_orb = 0
        for ish in range(self.n_shells):

            if self.compare_shells(shell=self.shells[ish],
                                   corr_shell=self.corr_shells[n_corr]):

                n_orb = self.corr_shells[n_corr]['dim']

                break

            else:

                offset += self.shells[ish]['dim']

        if n_orb == 0:

                self.report_error("Inconsistency between shells and corr_shells!")

        return n_orb, offset


    def _set_U_None(self):
        """
        Sets the following objects related to projections to None:

            * u_matrix

            * u_matrix_full

            * u_matrix_opt
        """

        self._u_matrix = None
        self._u_matrix_full = None
        self._u_matrix_opt = None


    def _set_proj_mat_dummy(self):
        """
        Redefines the shape of projection matrices in case dummy projections are used.
        In scenario  dummy projections are used number of MLWFs and Bloch states is assumed to be the same.
        """
        
        self.total_Bloch = self.total_MLWF
        self.proj_mat = numpy.zeros((self.n_k,
                                     self._n_spin,
                                     self.n_corr_shells, max([crsh['dim'] for crsh in self.corr_shells]),
                                     self.total_Bloch), numpy.complex_)

        self.n_orbitals = numpy.ones((self.n_k, self._n_spin), numpy.int_) * self.total_Bloch


    def _initialize_U_matrices(self):
        """

        Sets the following objects related to projections to empty numpy arrays:

            * u_matrix

            * u_matrix_full

            * u_matrix_opt

            * n_orbitals

        """

        self.n_orbitals = numpy.zeros((self.n_k, self._n_spin), numpy.int_)

        self._u_matrix_opt = numpy.zeros((self.total_Bloch, self.total_MLWF, self._n_spin, self.n_k), numpy.complex_)

        self._u_matrix = numpy.zeros((self.total_MLWF, self.total_MLWF, self._n_spin, self.n_k), numpy.complex_)

        self._u_matrix_full = numpy.zeros((self.n_k, self._n_spin, self.total_MLWF, self.total_Bloch), numpy.complex_)


    def _read_chkpt_fmt(self, filename=None):

        """
        Reads formatted checkpoint file from Wannier90 for one particular spin channel.
        Extracts U matrix.  In case there is disentanglement extracts also U_matrix_opt and ndimwin
        (ndimwin is how number of bands are called in Wannier90, in Wannier90Converter
        ndimwin is called n_orbitals in order  to be consistent with names from SumkDFT).

        :param filename:    Base name of file  with data
                            from Wannier90  for one particular spin channel
        :type filename:     str

        """
        err_msg = """Invalid  %s.chk.fmt! Dummy projections will be used in the calculation.""" % filename

        chkpt_data = {}
        dim = 3
        dim2 = 9


        try:
            if is_master_node():
                with open(name=filename + ".chk.fmt", mode="r") as input_file:
                    lines = input_file.readlines()
                    input_file.close()

                    pos = 0
                    msg = lines[pos].replace("\n", "")

                    # Prints out first line, remove space before dot comment line
                    self.make_verbose_statement("Checkpoint file has been opened. "
                                                "Checkpoint file was " + msg[:len(msg) - 1] + ".")

                    # num_bands
                    pos += 1
                    try:
                         chkpt_data["Number of bands"] = int(lines[pos])
                    except ValueError:
                        self.report_warning("Invalid number of bands in %s.chk.fmt!" % filename + err_msg)
                        self._produce_dummy_projections()
                        return

                    if self._total_Bloch_spin[self._name2SpinChannel[filename]] != chkpt_data["Number of bands"]:
                        self.report_warning("Different number of bands in "
                                            "%s.win  and in  %s.chk.fmt!" % (filename, filename) + err_msg)
                        self._produce_dummy_projections()
                        return

                    # number of excluded bands
                    pos += 1
                    try:
                         chkpt_data["Number of exclude bands"] = int(lines[pos])
                    except ValueError:
                        self.report_warning("Invalid number of excluded bands in %s.chk.fmt!" % filename + err_msg)
                        self._produce_dummy_projections()
                        return

                    for i in range(chkpt_data["Number of exclude bands"]):
                        pos += 1


                    temp_array = numpy.zeros((dim, dim), dtype=numpy.float_)
                    # Real lattice
                    pos += 1
                    try:
                        temp = lines[pos].split()
                        if len(temp) != dim2:
                            self.report_warning("Invalid real lattice matrix in %s.chk.fmt!" % filename + err_msg)
                            self._produce_dummy_projections()
                            return

                        for j in range(dim):
                            for i in range(dim):
                                temp_array[i, j] = float(temp.pop(0))
                        chkpt_data["Real lattice"] = temp_array

                    except ValueError:
                          self.report_warning("Invalid real lattice matrix in %s.chk.fmt!" % filename + err_msg)
                          self._produce_dummy_projections()
                          return

                    if not numpy.allclose(chkpt_data["Real lattice"], self._cell_cart):
                        self.report_warning("Real lattice matrix from  %s.chk.fmt differs from that "
                                            "found in %s.win file!" % (filename, filename) + err_msg)
                        self._produce_dummy_projections()
                        return

                    del temp_array

                    # Reciprocal lattice (not used by the converter)
                    pos += 1

                    # num_kpts
                    pos += 1
                    try:
                        chkpt_data["Number of kpoints"] = int(lines[pos])
                    except ValueError:
                        self.report_warning("Invalid number of kpoints in %s.chk.fmt!" % filename + err_msg)
                        self._produce_dummy_projections()
                        return

                    if self.n_k != chkpt_data["Number of kpoints"]:

                        self.report_warning("Different number of kpoints in"
                                            " %s.win  and in  %s.chk.fmt!" % (filename, filename) + err_msg)
                        self._produce_dummy_projections()
                        return


                    # mp_grid
                    pos += 1
                    temp = lines[pos].split()
                    try:
                        if len(temp) != dim:
                            self.report_warning("Invalid mp_grid in %s.chk.fmt!" % filename + err_msg)
                            self._produce_dummy_projections()
                            return

                        chkpt_data["mp_grid"] = []
                        for i in range(dim):
                             chkpt_data["mp_grid"].append(int(temp.pop(0)))
                    except ValueError:
                         self.report_warning("Invalid  mp_grid in %s.chk.fmt!" % filename + err_msg)
                         self._produce_dummy_projections()
                         return
                    del temp

                    if cmp(self.ham_nkpt, chkpt_data["mp_grid"]) != 0:

                        self.report_warning("Different k-point grid  in %s.win"
                                            "  and in  %s.chk.fmt!" % (filename, filename) + err_msg)
                        self._produce_dummy_projections()
                        return


                    # k points, not used by the converter
                    for i in range(chkpt_data["Number of kpoints"]):
                         pos += 1
                    # nntot, not used by the converter
                    pos += 1

                    # num wann
                    pos += 1
                    try:
                        chkpt_data["Number of Wannier orbitals"] = int(lines[pos])
                    except ValueError:
                        self.report_warning("Invalid number of Wannier orbitals in %s.chk.fmt!" % filename + err_msg)
                        self._produce_dummy_projections()
                        return

                    if self.total_MLWF != chkpt_data["Number of Wannier orbitals"]:
                        self.report_warning("Different number of Wannier orbitals  "
                                            "in %s.win  and in  %s.chk.fmt!" % (filename, filename) + err_msg)
                        self._produce_dummy_projections()
                        return

                    # checkpoint
                    pos += 1
                    chkpt_data["checkpoint"] = lines[pos].strip()

                    # have disentanglement
                    pos += 1
                    try:
                        idum = int(lines[pos])
                        if idum == 0 or idum == 1:
                            chkpt_data["have_disentanglement"] = (idum == 1)
                        else:
                            self.report_warning("Invalid have_disentanglement keyword in "
                                                " % s.chk.fmt, it should be 0 or 1!" % filename + err_msg)
                            self._produce_dummy_projections()
                            return
                    except ValueError:
                        self.report_warning("Invalid have_disentanglement in  %s.chk.fmt!" % filename + err_msg)
                        self._produce_dummy_projections()
                        return

                    if self._disentanglement_spin[self._name2SpinChannel[filename]] != chkpt_data["have_disentanglement"]:
                        self.report_warning("Have_disentanglement is different in " +
                                            " % s.win file and in %s.chk.fmt!" % (filename, filename) + err_msg)
                        self._produce_dummy_projections()
                        return


                    if chkpt_data["have_disentanglement"]:
                        # omega_invariant,  not used by the converter
                        pos += 1
                        # lwindow,   not used by the converter
                        for nkp in range(self.n_k):
                            for i in range(chkpt_data["Number of bands"]):
                                pos += 1

                        # ndimwin,  number of bands inside energy window
                        try:
                            for nkp in range(self.n_k):
                                pos += 1
                                self.n_orbitals[nkp, self._name2SpinChannel[filename]] = int(lines[pos])
                        except ValueError:
                            self.report_warning("Invalid number of bands "
                                                "inside energy window in  %s.chk.fmt!" % filename + err_msg)
                            self._produce_dummy_projections()
                            return

                        # U_matrix_opt
                        try:
                            for nkp in range(self.n_k):
                                for j in range(chkpt_data["Number of Wannier orbitals"]):
                                    for i in range(chkpt_data["Number of bands"]):
                                        pos += 1
                                        temp = lines[pos].split()  # two entries per line are expected real, imag
                                        self._u_matrix_opt[i, j, self._name2SpinChannel[filename], nkp] = float(temp[0]) + 1j * float(temp[1])
                        except ValueError:
                            self.report_warning("Invalid value of matrix U_opt!" + err_msg)
                            self._produce_dummy_projections()
                            return
                        
                        if self.SO:

                            self._rearrange_U(U_mat=self._u_matrix_opt, filename=filename)

                    else:
                        for nkp in range(self.n_k):
                            self.n_orbitals[nkp, self._name2SpinChannel[filename]] = self.total_MLWF

                    # U_matrix
                    try:
                        for nkp in range(self.n_k):
                            for j in range(chkpt_data["Number of Wannier orbitals"]):
                                for i in range(chkpt_data["Number of Wannier orbitals"]):
                                    pos += 1
                                    temp = lines[pos].split()  # two entries per line are expected real, imag
                                    self._u_matrix[i, j, self._name2SpinChannel[filename], nkp] = float(temp[0]) + 1j * float(temp[1])
                    except ValueError:
                        self.report_warning("Invalid value of matrix U!" + err_msg)
                        self._produce_dummy_projections()
                        return

                    if self.SO:

                       self._rearrange_U(U_mat=self._u_matrix, filename=filename)


        except IOError:
            self.report_warning("Opening file %s.chk.fmt failed. " % filename + err_msg)
            self._produce_dummy_projections()
            return

        self.make_statement("The information from formatted %s.chk.fmt file has been read." % filename)

        return chkpt_data


    def _rearrange_U(self, U_mat=None, filename=None):
        """
        Rearranges content of U_mat. To be used in case of spin-orbit coupling.
        :param U_mat: U matrix to rearrange

        """
        # in case of spin-orbit coupling we have to rearrange:
        # 1_up 1_down 2_up 2_down ..... n_up n_down   to
        # 1_up 2_up ....n_up 1_down 2_down ..... n_down

        num_wann_channel = self.total_MLWF / 2
        ind = range(0, self.total_MLWF, 2)
        temp_array = deepcopy(U_mat)
        for nkp in range(self.n_k):
            for ii, ind_i in enumerate(ind):
                for jj, ind_j in enumerate(ind):
                    U_mat[ii, jj, self._name2SpinChannel[filename], nkp] = temp_array[ind_i, ind_j, self._name2SpinChannel[filename], nkp]
                    U_mat[ii + num_wann_channel, jj + num_wann_channel, self._name2SpinChannel[filename], nkp] = temp_array[ind_i + 1, ind_j + 1, self._name2SpinChannel[filename], nkp]




    def _produce_full_U_matrix(self):
        """
        Produces full rotation matrix from Bloch states to MLWFs.
        """

        temp_array = numpy.zeros((self.total_Bloch, self.total_MLWF), numpy.complex_)


        if self._u_matrix is None:
            self.report_error("U_matrix is not set!")
        elif any(self._disentanglement_spin) and self._u_matrix_opt is None:
            self.report_error("U_matrix_opt is not set")

        for k in range(self.n_k):
            for spin_bloc in range(self._n_spin):

                if self._disentanglement_spin[spin_bloc]:
                    temp_array = numpy.dot(self._u_matrix_opt[:, :, spin_bloc, k], self._u_matrix[:, :, spin_bloc, k])
                else:
                    temp_array = self._u_matrix[:, :, spin_bloc, k]

                # convention in Wannier90: O' = U^{\dagger} OU
                # where O in Bloch basis and O' in MLWFs basis
                #
                # convention in Wannier90Converter
                # O' = U O U ^{\dagger}
                # where O in Bloch basis and O' in MLWFs basis
                #
                # we have to transform
                # U_matrix_full -> U_matrix_full^{\dagger}
                # for each spin channel and k-point

                self._u_matrix_full[k, spin_bloc, :, :] = temp_array.conjugate().transpose()


    def _produce_projections(self):
        """
        In case correlated and non correlated MLWFs in the system dummy projections will be built.
        Projections from Bloch to correlated space are built otherwise.

        Produces projections required by SumkDFT, in case *.chk.fmt file(s) is/are present.
        Dummy projections in the form of identity matrices are produced otherwise.

        Projections are produced for all correlated shells, also those which are symmetry equivalent.
        Projections have a form of rotation from Bloch space to the correlated subspace.

        """

        # If number of shells is larger than number of correlated shells  then seedname*.chk
        # is neglected and dummy projections are always built.

        if len(self.shells) > len(self.corr_shells):
            if self._n_spin == 1:
                self.make_verbose_statement("Number of shells larger than the number of "
                                            "correlated shells. %s.chk.fmt file will be" % self._filename +
                                            " neglected and dummy projections will be built.")
            elif self._n_spin == 2:
                self.make_verbose_statement("Number of shells larger than the number of "
                                            "correlated shells. %s_up.chk.fmt "
                                            "and %s_down.chk.fmt files will be" % (self._filename, self._filename) +
                                            " neglected and dummy projections will be built.")
            else:
                self.report_error("Improper value of 'n_spin'. Method '_produce_projections' "
                                  "should be executed after data from *win file(s) is read!")

            self._produce_dummy_projections()
            return

        self.proj_mat = None
        self._initialize_U_matrices()

        if is_master_node():

            # case of a system with only correlated MLWFs
            if self._n_spin == 1:

                self._read_chkpt_fmt(filename=self._filename)
                # checks if self._proj_mat is not already dummy projections
                if self.proj_mat is not None:

                    return

            elif self._n_spin == 2:
                # up channel
                self._read_chkpt_fmt(filename=self._filename + "_up")
                # checks if self._proj_mat is not already dummy projections
                if self.proj_mat is not None:

                    return

                else:

                    self._read_chkpt_fmt(filename=self._filename + "_down")

                # checks if self._proj_mat is not already dummy projections
                if self.proj_mat is not None:

                     return

            else:
                self.report_error("Invalid number of spin blocs!")

            max_num_band = max(max(self.n_orbitals))
            if self.total_Bloch != max_num_band:
                self.report_error("Number of Bloch states is different than the maximum "
                                  "number of bands in the energy window used for "
                                  "the construction of MLWFs! "
                                  "Number of Bloch states is %s " % self.total_Bloch +
                                  "Maximum number of bands in the energy window is %s" % max_num_band)

            self._produce_full_U_matrix()

            # Initialises the projections:
            self.proj_mat = numpy.zeros((self.n_k, self._n_spin, self.n_corr_shells, max([crsh['dim'] for crsh in self.corr_shells]),
                                        self.total_Bloch), numpy.complex_)

            for icrsh in range(self.n_corr_shells):
                n_orb, offset = self.eval_offset(n_corr=icrsh)

                for ik in range(self.n_k):
                    for isp in range(self._n_spin):
                        self.proj_mat[ik, isp, icrsh, 0: n_orb, :] = self._u_matrix_full[ik, isp, offset: offset + n_orb, :]

        if self._n_spin == 1:
            self.make_statement("Projections  based on the information "
                                "from %s.chk.fmt file have been built." % self._filename)
        elif self._n_spin == 2:
            self.make_statement("Projections  based on the information from"
                                " %s_up.chk.fmt and %s_down.chk.fmt "
                                "files  have been built." % (self._filename, self._filename))


    def _produce_dummy_projections(self):
        """
        Produces dummy projections.
        """
        self._set_U_None()
        self._set_proj_mat_dummy()
        self._produce_projections_dummy_core()
        self._dummy_projections_used = True
        self.make_statement("Dummy projections have been built.")


    def _produce_projections_dummy_core(self):
        """"

        Produces dummy projections ("rectangular identity matrices")
        required by SumkDFT in case chk.fmt file is not present or not used.

        """

        for icrsh in range(self.n_corr_shells):
            n_orb, offset = self.eval_offset(n_corr=icrsh)
            for ik in range(self.n_k):
                for isp in range(self._n_spin):
                    self.proj_mat[ik, isp, icrsh, 0: n_orb, offset:offset + n_orb] = numpy.identity(n_orb)


    def _produce_hopping(self):
        """

        Produces hopping integrals (rotated from Bloch space to MLWFs).

        """

        self.hopping = numpy.zeros((self.n_k, self._n_spin, self.total_Bloch, self.total_Bloch), numpy.complex_)

        if self._dummy_projections_used:
            for k in range(self.n_k):
                for s in range(self._n_spin):

                        # transformation from list to numpy.ndarray
                        self.hopping[k, s] = self.H_k[k][s][:, :]

        else:
            for k in range(self.n_k):
                for s in range(self._n_spin):

                        # rotate from MLWF to Bloch to get hopping
                        self.hopping[k, s] = numpy.dot(self._u_matrix_full[k, s].conjugate().transpose(),
                                                       numpy.dot(self.H_k[k][s], self._u_matrix_full[k, s]))


        self.make_statement("Hopping integrals have been evaluated.")



    def _set_k_dep_projection(self):
        """
        Sets k_dep_projection which is required by SumkDFT.
        """
        if any(self._disentanglement_spin[indx] for indx in range(self._n_spin)):
            self.k_dep_projection = 1
        else:
            self.k_dep_projection = 0


    def _SumkDFT_par(self):
        """
        Prepares input parameters for SumkDFT in the form of a Python dictionary.
        **Parameters**::

        -  energy_unit (float): the unit of energy used in the calculations,
           it is fixed to energy_unit=1.0 (always assuming eV units)

        -  n_k (int): number of k-points used in the summation. This is evaluated from mp_grid keyword
           which is read from Wannier90 input file (*.win).

        -  k_dep_projection (int): whether or not dimension of projection operator depends on the k-point:

            * 0 dimension of projection operator  does not depend on k-points,

            * 1 dimension of projection operator  depends on k-points.

            In case of disentanglement for non spin-polarised case or spin-orbit coupling case this parameter
            is set to 1, and it is set to 0 otherwise.

            In case of spin-polarised calculation if there is disentanglement
            for any spin channel this parameter is set to 1, and it is set to 0 otherwise.


        - SP (int):	defines whether we have spin-polarised or spin-less calculation. \n
                Possible values:

                * SP=0 spin-less calculations,

                * SP=1 spin-polarised calculation.

                In case seedname_up.win and seedname_down.win files are found in
                the working directory this parameter is set to 1.

                In case spinors keyword is found in seedname.win this parameter is set to 1.

                Otherwise this parameter is set to 0 and spinless calculation is assumed.

        - SO (int): defines whether or not we have spin-orbit calculations.\n
            Possible values are:

                * SO=0   no spin-orbit calculations

                * SO=1   spin-orbit calculations

                In case a keyword spinors = true  or  spinors = t (value of the keyword is case independent)
                found in seedname.win  SO = 1,  and  it is 0 otherwise.

        - charge_below (float): the number of electrons in the unit cell below
          energy window, *it is set to 0 in Wannier90Converter*

        - density_required (float):

                Used in the search for the chemical potential.

                This is the total number of electrons in the Hamiltonian expressed in MLWF basis
                (correlated and non-correlated electrons).

                This parameter is calculated using fermi_energy keyword from
                case.win file and H_k. For each k-point H_k is diagonalized and states which are
                below or equal Fermi level are counted. The counter divided by
                the number of k-points gives a required density.

        - symm_op (int): defines if symmetry operations are used for the BZ sums\n.

                 Possible values are:

                 *   symm_op=0 all k-points explicitly included in calculations,

                 *   symm_op=1 symmetry inequivalent points used, and rest k-points
                     reproduced by means of the symmetry operations.

                 This parameter is fixed to symm_op=0 for Wannier90Converter (no symmetry groups for the k-sum).

        - n_shells (int): number of all sites in the unit cell

        - shells (list of dictionaries with values are  int): first dimension is n_shells, second dimension is 4,
             [{"atom":index, "sort":sort, "l":l, "dim":dim },{...}]. Keywords in each entry are as follows: \n

             * index: number of the atom

             * sort: label which marks symmetry equivalent atoms

             * l: angular momentum quantum number

             * dim: number of orbitals for the particular shell

             A brief description of shells determination (in two steps):

                    Step 1
                For each shell index, l, and dim are evaluated basing on the information from case.win file.
                All three attributes are determined from the definition of initial projections. Here the assumption is
                made that initial projections are chosen reasonably. For example if final MLWFs have d-like
                shape d atomic orbitals should be defined as initial projections. If for example for some reason
                a user would define p-like initial projections and obtain d-like MLWFs than Wannier90Converter may
                produce wrong input data for SumkDFT.

                    Step 2
                Next, sort is calculated. It is evaluated from  H_R. An array of H_R for R=0 (ham_r parameter) is used
                in the construction. Since the dimension of each shell is known from the previous step for every shell a
                sub-block of the array which corresponds to that shell can be determined. The order of initial
                projections from case.win file maps onto the order of sub-blocks in the array. Each of such a sub-block
                is diagonalized and eigenvalues of sub-blocks are compared (eigenvalues are sorted before a comparison).
                If a sub-block has the same eigenvalues as the previously examined n-th sub-block then it has the same
                sort as the n-th shell to which this n-th sub-block corresponds. In case eigenvalues of the currently
                examined sub-block are different from  eigenvalues of all so far examined sub-blocks then a shell which
                corresponds to this sub-block has a new unique sort. This procedure is continued until all shells have
                their sort parameter set.


        - n_corr_shells (int): number of all correlated sites in the unit cell

        - corr_shells (list of dictionaries which values are int): first dimension is n_corr_shells,
          second dimension is 6, [{"atom":index, "sort":sort, "l":l, "dim":dim ,"SO":SO, "irep":irep].
          Keywords in each entry are as follows:\n

             * index: number of the atom

             * sort: label which marks symmetry equivalent atoms

             * l: angular momentum quantum number

             * dim: number of orbitals for the particular shell

             * SO: spin-orbit coupling, if it is enabled it is 1, and it is 0 otherwise.

              In case a keyword spinors = true  or  spinors = t (value of keyword is case independent)
              found in seedname.win  SO = 1, and it is 0  otherwise.


             * irep: dummy parameter is set to 0


              Two scenarios can be possible.

                User-defined correlated shells:
                  In this scenario user puts non_standard_corr_shells=true to the file with the extra parameters.
                  In that case correlated shells are the same as shells (they are the same apart from attributes
                  SO and irep). This feature is provided in case a user needs to treat shells usually considered as
                  non correlated as correlated. This feature can be used if for example a user wants to treat p
                  orbitals as correlated orbitals.

                Correlated shells evaluated from shells:
                  In this scenario shells (after sort attribute is evaluated) are examined.
                  All those shells for which l is equal or larger than 2  are used to build corr_shells. For all
                  correlated shells parameter irep is set to 0. If spinors keyword was found in input win file then for
                  each correlated shell SO attribute is set to 1. Otherwise attribute SO is set to 0.

        - use_rotations (int): if local of global coordinates are used,
          *This parameter is fixed to use_rotation=1 for Wannier90Converter*

        - rot_mat (numpy.array.complex): if use_rotation=1, then this contains a list of the rotation matrices.
          *This parameter is enabled for Wannier90Converter*

          In order to find rotation matrices H_R for R=0 (ham_r parameter) is examined. Sub-blocks which correspond
          to correlated shells are diagonalized and their eigenvectors are used to produce rotation matrices
          between inequivalent correlated shells and shells related to them by symmetry.

        - rot_mat_time_inv (list of int): this is used only if  use_rotations=1
          *This parameter is disabled for Wannier90Converter*

          This parameter is fixed to 0 in Wannier90Converter for all shells.

        - n_reps (int): Number of irreducible representations of the correlated shells,
            for example for t2g/eg splitting it wil be 2
            *This parameter is not used  by  Wannier90Converter*

            This parameter is set to -1.

        - dim_reps (list of int): Dimension of representation (for example for t2g/eg [2,3])
          *This parameter is not used  by  Wannier90Converter*

          This parameter is set to -1.

        - T (list of numpy.array.complex):

           Each element in the list corresponds
           to the particular inequivalent correlated shell. Each element in the list is a transformation matrix
           from the spherical harmonics to cubic basis.

           T matrix is evaluated basing on the information about the initial projections from case.win file.
           The exact form of the T matrix is determined by the order of the initial projections from case.win file.

           In case of spin-orbit coupling scenario T matrix is constructed according to  "non-mixing" base scenario.

           In case a number of orbitals for a shell is smaller than the maximal number of orbitals for that shell a
           generic T matrix is produced using spherical_to_cubic method (in that case the content of the matrix
           does not depend on the order of the initial projections).

        -n_orbitals(numpy.array.int, dimension n_k, SP+1-SO,n_corr_shells,max(corr_shell dimension)):
         total number of orbitals for each k-point.

           A brief description of n_orbitals determination:

            In case there is no need for the disentanglement procedure during the construction of MLWFs each entry in
            n_orbitals is equal to the total number of MLWFs (keyword num_wann from case.win file).

            In case of the disentanglement procedure during the construction of MLWFs content of
            n_orbitals is read from the formatted checkpoint file (case.chk.fmt).

        -proj_mat(numpy.array.complex, dimension [n_k, SP+1-SO,n_corr_shells,max(corr_shell dimension)]):
         U matrix which transforms pseudo-Bloch states to correlated MLWF.

           A brief description of proj_mat determination:

             In case formatted checkpoint file: case.chk.fmt is provided proj_mat are constructed from
             U matrices which are read from it.

             In case formatted checkpoint is not found "dummy" projections are built.

             In case there are non-correlated shells in the Hamiltonian (number of shells larger than the number
             of correlated shells) case.chk.fmt is neglected and "dummy" projections are built.

        -bz_weights (numpy.array.float): one dimensional array which
        contains weights of the k-points for the summation.

           Since no symmetry of FBZ is used all elements of this
           list have the same value: 1/n_k.

        -hopping(numpy.array.complex, dimension [n_k, SP+1-SO, Max(n_orbitals), Max(n_orbitals)]):
        non-interacting H_k at each k-point.

            A brief description of hopping determination:

               In case formatted checkpoint is used to built projections hopping is
               constructed from rotation of H_k with projections as rotating matrices.

               In case projections are "dummy" projections H_k becomes the hopping array.

        """

        bz_weights = numpy.ones(self.n_k, numpy.float_) / float(self.n_k)

        self._set_k_dep_projection()

        self._SumkDFT_data = {"energy_unit": 1.0,
                              "n_k": self.n_k,
                              "k_dep_projection": self.k_dep_projection,
                              "SP": self.SP,
                              "SO": self.SO,
                              "charge_below": 0,
                              "density_required": self.density_required,
                              "symm_op": 0,
                              "n_shells": self.n_shells,
                              "shells": self.shells,
                              "n_corr_shells": self.n_corr_shells,
                              "corr_shells": self.corr_shells,
                              "use_rotations": 1,
                              "rot_mat": self.rot_mat,
                              "rot_mat_time_inv": [0 for i in range(self.n_shells)],
                              "n_reps": -1,  # not used
                              "dim_reps": -1,  # not used
                              "T": self.T,
                              "n_orbitals": self.n_orbitals,
                              "proj_mat": self.proj_mat,
                              "bz_weights": bz_weights,
                              "hopping": self.hopping,
                              "n_inequiv_shells": self.n_inequiv_corr_shells,
                              "corr_to_inequiv": self.corr_to_inequiv,
                              "inequiv_to_corr": self.inequiv_to_corr,
                              "chemical_potential": self.chemical_potential}


    # getters
    def R_sym(self, n_corr):

        """
        :param n_corr: number of correlated shell
        :type n_corr: int

        :return: Symmetry operation for the particular correlated shell in the following format :
                R_sym={"atom":index, "sort":sort, "rot_mat":rot_mat, "eig":eig}

                Description:
                    * index: index of atom

                    * sort: label which marks symmetry equivalent atoms,
                            it is the same for the symmetry equivalent atoms

                    * rot_mat: symmetry operation for the particular n_corr correlated shell

                    * eig: eigenvalues of Hamiltonian block which
                    correspond to the particular symmetry operation

        """
        if self.check_inequivalent_corr(n_corr=n_corr):
            return self._R_sym[n_corr]
        else:
            return None


    @property
    def U_matrix(self):
        """

        :return: Rotation matrix from smooth pseudo-Bloch states to MLWF.
                 In case there is no disentanglement returns rotation matrix from Bloch states to MLWF.
        """
        return self._u_matrix


    @U_matrix.setter
    def U_matrix(self, value=None):
        """
        
        :param value: Potential new value
        :return:
        """
        self.report_error("Value of U_matrix cannot be overwritten!")


    @property
    def U_matrix_opt(self):
        """

        :return:    In case there is disentanglement rotation from
                    Bloch states to smooth pseudo-Bloch states, and  None otherwise.

        """
        return self._u_matrix_opt


    @U_matrix_opt.setter
    def U_matrix_opt(self, value=None):
        """

        :param value: Potential new value
        """

        self.report_error("Value of U_opt matrix cannot be overwritten!")


    @property
    def U_matrix_full(self):
        """

        :return: Whole rotation matrix from Bloch states to MLWF.
        If dummy projections are used it is set to None.

        """

        return self._u_matrix_full


    @U_matrix_full.setter
    def U_matrix_full(self, value=None):
        """

        :param value: Potential new value
        """
        self.report_error("Value of U_full matrix cannot be overwritten!")


    @property
    def SumkDFT_input_folder(self):
        """

            :return: Folder from hdf file in which SumkDFT data is stored
            :rtype : str

        """
        return self._SumkDFT


    @SumkDFT_input_folder.setter
    def SumkDFT_input_folder(self, value=None):
        """
        Prevents user from overwriting SumkDFT_input_folder
        :param value: new potential value
        :type value: str
        """
        self.report_error("Attribute SumkDFT_input_folder cannot be modified by user!")


    def get_hopping(self, n_k, n_spin_bloc):
        """

        :param n_k: n_k-th k-point
        :type n_k: int

        :param n_spin_bloc: number of spin-bloc  (in case of spinless calculation we have only one spin-block
                            and valid value is  0, in case of spin-polarised we have two blocs
                            and valid values are 0,1)
        :type n_spin_bloc: int

        :return: If valid input parameters provided returns hopping integrals
                 in the following format for the particular k-point and spin-bloc.
                 If input parameters are invalid returns None
        :rtype :numpy.array.complex, dimension [ Max(n_orbitals), Max(n_orbitals)]

        """
        if self.check_n_k(n_k=n_k) and self.check_n_spin_bloc(n_spin_bloc=n_spin_bloc):
            return self.hopping[n_k, n_spin_bloc, :, :]
        else:
            return None


    def get_projector(self, n_k, n_spin_bloc, n_corr_shell):
        """
        :param n_k: n-th k-point
        :type n_k: int

        :param n_spin_bloc: number of spin-bloc  (in case of spinless calculation we have only one spin-block
                            and valid value is  0, in case of spin-polarised we have two blocs
                            and valid values are 0,1)
        :type n_spin_bloc: int

        :param n_corr_shell: number of correlated shell
        :type n_corr_shell: int

        :return: if valid input parameters provided returns projector array, and  return None otherwise.
        :rtype: numpy.array.complex, dimension [Max(corr_shell dimensions), Max(n_orbitals)]]

        """

        if (
           self.check_n_k(n_k=n_k) and
           self.check_n_spin_bloc(n_spin_bloc=n_spin_bloc) and
           self.check_inequivalent_corr(n_corr=n_corr_shell)):

            return self.proj_mat[n_k, n_spin_bloc, n_corr_shell, :, :]

        else:

            return None


    @property
    def all_read(self):
        """

        :return: True if all items necessary for Wannier90Converter
        object to function properly were loaded, and  False otherwise.
        :rtype : bool

        """
        return self._all_read

    @all_read.setter
    def all_read(self, value=None):
        """

        :param value:  Potential new value
        :return:
        """
        self.report_error("Attribute all_read cannot be modified by user!")





