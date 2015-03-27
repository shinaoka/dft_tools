import numpy
import math
import cStringIO
from os.path import isfile
from os import  remove
from copy import deepcopy

# pytriqs
from pytriqs.utility import mpi
from pytriqs.archive import HDFArchive
from pytriqs.applications.dft.U_matrix import  spherical_to_cubic
from pytriqs.applications.dft.converters.converter_tools import ConverterTools
from pytriqs.applications.dft.messaging import Check, Save
from pytriqs.applications.dft.wannier_converter_dataIO import AsciiIO


class Wannier90Converter(Check,ConverterTools,Save):
    """
        *Class for lattice Green's function objects*

        **Functionality**::

            (main methods mentioned only please have a look below for all methods)

            -  Reads H_R from wannier90.x output file, Fourier transform it to H_k.
                (method: __h_to_triqs)

            -  Reads projectors from seedname*.chk.fmt file(s) and transforms it to the proper format.
               seedname*.chk.fmt are formatted checkpoints files, (formatted files store data in machine independent way)
                (method: read_format_chkpt)

            -  Calculates hopping integrals
                (method: _produce_hopping)

            -  Calculates symmetry operations which transforms each inequivalent site
                to all other sites related to that site by symmetry
                (method: __h_to_triqs)

            -  Prepares parameters necessary for sumk_dft
                (method: _sumk_dft_par)

            -  Writes input data for sumk_dft into hdf file
                (inherited method: method _save_par_hdf)

            -  Writes Wannier2TRIQS converter parameters into hdf file
                (inherited method: method _save_par_hdf)

            -  Checks if input parameters for sumk_dft have changed since the last run

                (inherited method: parameters_changed)

            -  Checks if parameters for Wannier2TRIQS converter have changed since the last run
            `
                (inherited method: parameters_changed)


        **Input files for the calculation (to change):**

            Number of files needed by converter and their names depend on polarisation scenario:

                -   Non-polarised calculation:

                        *seedname.win (obligatory)

                        *seedname.chk (optional)

                        *seedname_hr.dat (obligatory)

                -   Spinpolarised calculation:

                        * seedname_up.win, seedname_down.win (obligatory)

                    For each spin channel calculation is performed
                    separately and as a result we get:

                        * seedname_up.chk, seedname_down.chk, (optional)

                        * seedname_hr_up.dat, seedname_hr_down.dat (obligatory)

                -   Spin-orbit-coupled  calculation:

                        *seedname.win (obligatory)

                        *seedname.chk (optional)

                        *seedname_hr.dat (obligatory)

                    Keyword spinors = true in seedname.win is obligatory in this case  and
                    only half of the initial projections should be provided.

            - If seedname*.chk are not present, dummy projectors in the
              form of rectangular matrices each with block of identity matrix
              which corresponds to the given correlated shell will be calculated,

            - If number of shells is larger than number of correlated shells  then seedname*.chk
              is neglected and dummy projectors are always built,

            - Otherwise projectors basing on data from  seedname*.chk will be constructed.
              In case seedname*.chk is present check for consistency
              between seedname*.win and seedname*.chk will be done.

            In case dummy projectors are built, Hk becomes  hopping integrals.

            If there is no *seedname_hr.dat calculation will not start.


    """
    #static variables:
    # mapping of booleans from Fortran to Python
    _fortran_boolean={"False":False,"True":True}
    corr_shells_keywords=["atom","sort","l","dim","SO","irep"]
    shells_keywords=["atom","sort","l","dim"]

    def __init__(self, filename=None, extra_par_file=None, dft_subgrp="SumK_DFT"):

        """Constructor for Wannier90Converter object.

            :param filename:  			name of file with H_R: filename_hr.dat, DMFT data.
                                        will be stored  in filename.h5
            :type filename: 			str

            :param extra_par_file:      text file with some extra parameters (num_zero, verbosity).
                                        If not set then default value will be used,
            :type  extra_par_file:      str

            :param dft_subgrp:			Name of the directory in filename.h5 where input
                                        parameters for sumk_dft  are stored
            :type dft_subgrp:			str


            """

        super(Wannier90Converter, self).__init__()

        #********************* Sorted fields **********************************


        self.FULL_H_R = None # H_R

        self.Hk = None # H_k

        self.SO = None # defines whether or not we have a spin-orbit calculation

        self.SP = None # defines whether or not we have a spin polarised  calculation

        self.T = None # Rotation matrices: complex harmonics to cubic harmonics for each inequivalent correlated shell

        self.Vector_R_degeneracy = None # degeneracy of each Wigner-Seitz grid point

        self.Vector_R = None # vector with Wigner-Seitz grid points for Fourier transformation H_R -> H_k

        self.chemical_potential=None # initial guess for the chemical potential

        self.corr_shells = None # all correlated shells (also symmetry equivalent)

        self.corr_to_inequiv = None # mapping: correlated shell -> inequivalent correlated shell

        self.density_required = None # total number of electrons

        self.k_point_mesh = None # k-point grid

        self.inequiv_to_corr = None # mapping: inequivalent correlated shell -> correlated shell

        self.n_corr_shells = None # number of all correlated shells (both inequivalent and equivalent)

        self.n_k = None # total number of k-points

        self.n_inequiv_corr_shells = None # number of inequivalent correlated shells

        self.n_orbitals = None # number of bands used to produce MLWF for each spin channel and each k-point

        self.n_shells= None  # number of all shells (both correlated and non-correlated)

        self.proj_mat = None # projectors for DMFT calculations

        self.rot_mat = None # rotation matrices for symmetry equivalent sites

        self.shells = None # all shells (also symmetry equivalent)

        self.total_Bloch = None # total number of Bloch states used for the construction of MLWF

        self.total_MLWF = None # total number of MLWF (both correlated and an correlated)


        self._R_sym = None # symmetry operations

        # data structure with all states implemented in Wannier 90 which are used as an initial guess for MLWF,
        # each entry in the list has the following entries
        # l   angular momentum, (but for hybridisation it just  a label with negative value)
        # mr  number of state for the particular l
        # name label of state or group of states
        # dim -- number of states for the particular initial guess
        # corr -- correlated or not correlated

        # labels s,p, etc. for states taken from Wannier90 user guide
        # we want to project on orbitals with pure correlated or non-correlated character,
        # orbitals like sp3d are not valid

        self._all_states=[{"l":0,"mr":1,"name":"s","dim":1},

                    {"l":1,"mr":None,"name":"p","dim":3},
                    {"l":1,"mr":1,"name":"pz","dim":1},
                    {"l":1,"mr":2,"name":"px","dim":1},
                    {"l":1,"mr":3,"name":"py","dim":1},

                    {"l":2,"mr":None,"name":"d","dim":5},
                    {"l":2,"mr":1,"name":"dz2","dim":1},
                    {"l":2,"mr":2,"name":"dxz","dim":1},
                    {"l":2,"mr":3,"name":"dyz","dim":1},
                    {"l":2,"mr":4,"name":"dx2-y2","dim":1},
                    {"l":2,"mr":5,"name":"dxy","dim":1},

                    {"l":3,"mr":None,"name":"f","dim":7},
                    {"l":3,"mr":1,"name":"fz3","dim":1},
                    {"l":3,"mr":2,"name":"fxz2","dim":1},
                    {"l":3,"mr":3,"name":"fyz2","dim":1},
                    {"l":3,"mr":4,"name":"fxyz","dim":1},
                    {"l":3,"mr":5,"name":"fz(x2-y2)","dim":1},
                    {"l":3,"mr":6,"name":"fx(x2-3y2)","dim":1},
                    {"l":3,"mr":7,"name":"fy(3x2-y2)","dim":1},

                    {"l":-1,"mr":None,"name":"sp","dim":2},
                    {"l":-1,"mr":1,"name":"sp-1","dim":1},
                    {"l":-1,"mr":2,"name":"sp-2","dim":1},

                    {"l":-2,"mr":None,"name":"sp2","dim":3},
                    {"l":-2,"mr":1,"name":"sp2-1","dim":1},
                    {"l":-2,"mr":2,"name":"sp2-2","dim":1},
                    {"l":-2,"mr":3,"name":"sp2-3","dim":1},

                    {"l":-3,"mr":None,"name":"sp3","dim":4},
                    {"l":-3,"mr":1,"name":"sp3-1","dim":1},
                    {"l":-3,"mr":2,"name":"sp3-2","dim":1},
                    {"l":-3,"mr":3,"name":"sp3-3","dim":1},
                    {"l":-3,"mr":4,"name":"sp3-4","dim":1}
                    ]

        self._all_read = None # whether or not data needed in rerun is in the hdf file

        self._atoms= None # stores all projections read so far from win file

        self._disentangled_spin = None # defines disentanglement separately for each spin

        self._dummy_projectors_used = False # defines whether or not dummy projectors are in use

        self._extra_par={"num_zero":0.0001,"verbosity":2} # definition of parameters from extra_par_file

        self._extra_par_file = None # name of file with some extra parameters (num_zero, verbosity, T)

        self._filename = None # core name of important files: filename_hr.dat,
        #                       filename.h5, filename.win, filename_up.win,
        #                       filename_down.win, filename.chk.fmt,
        #                       filename_up.chk.fmt, filename_down.chk.fmt

        self._win_par={ "fermi_energy": None, # definition of parameters from win file needed by converter
                        "mp_grid": None,
                        "hr_plot":None,
                        "spinors":False,
                        "projections":None}

        self._name2SpinChannel = None  # defines mapping between name of file and spin channel

        self._num_zero = None  # definition of numerical zero

        self._n_spin = None # defines spin dimension for DFT data,
        #                     it is 1 for spin non-polarised, spin-orbit calculation
        #                     and 2 for spin polarised calculation

        self._parameters = None  # parameters specific for Wannier converter

        self._sumk_dft = None # folder in hdf file where dft input data is stored (both sumkdft and wannier converter)

        self._u_matrix = None  # transformation from (pseudo)-Bloch to MLWF

        self._u_matrix_full = None # transformation form Bloch to MLWF

        self._u_matrix_opt = None # transformation form Bloch to pseudo-Bloch smooth states

        self._verbosity = None # defines level of output verbosity

        self.ham_nkpt = None # vector with three elements which defines k-point grid


        if  isinstance(filename, str):
            self._filename = filename
        else:
            self.report_error("Give a string for keyword 'filename' so that input "
                              "Hamiltonian can be read from filename_hr.dat and "
                              "DMFT calculations can be saved into filename.h5!")

        if  isinstance(extra_par_file, str):
            self._extra_par_file = extra_par_file
        elif extra_par_file is None:
            self.make_verbose_statement("Default values will be used")
        else:
            self.report_error("Give a string for keyword 'extra_par_file' so that additional  "
                              "parameters can be read!")

        if  isinstance(dft_subgrp, str):
            self._sumk_dft = dft_subgrp
        else:
            self.report_error(""""Give a name for a folder in hdf5 in
                              which results from sumk_dft will be kept!""")


    def convert_dft_input(self,corr_shells=None):
        """
            Reads the input files, and stores the data in the HDF file. Checks if we have rerun or fresh calculation,
            In case of rerun checks if parameters changed (if crucial parameters changed it will stop).


             **Parameters**::
                * Parameters read from the extra_input,txt text file

                    - verbosity: if verbosity=2 then additional information  is printed
                                 out to the standard output, in particular additional information
                                 will be printed in case input parameters have changed from the last time.
                                 verbosity type : int

                    - num_zero: defines what we mean by numerical zero, it is used to check if the structure of
                                Hamiltonian is correct
                                num_zero type : float

                In case of parameters read from extra_input,txt case of each letter in every
                keyword is important. In case there is no such file default values will be used.

                * Parameters extracted from data stored in filename.win file

                    - density_required: total number of electrons (number of correlated electrons and non correlated
                                        electrons in the whole system described by lattice Hamiltonian in MLWF basis.

                                        This parameter is calculated from Hk (Fourier transformed HR) and  fermi_energy
                                        keyword from filename.win

                                        density_required type : float


                    -corr_shells:   Defines structure of correlated shells. Every list represents one correlated site:\n
                                    Keywords in each entry are as follows:
                                    [{"atom":atom, "sort":sort, "l":l, "dim":dim, "SO":SO, "irep":irep }, {....}]\n
                                    Description:

                                        * atom: number of the atom

                                        * sort: label which marks symmetry equivalent atoms,
                                                it is the same for the symmetry equivalent atoms

                                        * l: angular momentum quantum number

                                        * dim: number of orbitals for the particular shell

                                        * SO: spin orbit, if included is 1 otherwise it is 0)

                                        * irep: dummy parameter set to 0

                                    Parameter corr_shells is calculated form the following block:

                                       Begin Projections
                                        ---------------
                                        ---------------
                                        ---------------
                                       End Projections

                                    in filename.win

                                    Parameter corr_shells is used to calculate T matrix. T is
                                    transformation matrix from spherical to real harmonics needed for
                                    systems which require Hamiltonian beyond Kanamori Hamiltonian,
                                    T has a form of the list, every element of the list T corresponds
                                    to the different inequivalent correlated shell,
                                    every element of the list is of type dict.

                                    corr_shells type : list of dictionaries which values are int

                    - shells:   Defines structure of all shells. Every list represents one site.
                                Keywords in each entry are as follows: \n
                                [{"atom":atom, "sort":sort,"l":l, "dim":dim, }, {....}]
                                Description:

                                    * atom: number of the atom

                                    * sort: label which marks symmetry equivalent atoms,
                                            it is the same for the symmetry equivalent atoms

                                    * l: angular momentum quantum number

                                    * dim: number of orbitals for the particular shell

                                Parameter shells is calculated form the following block:

                                       Begin Projections
                                        ---------------
                                        ---------------
                                        ---------------
                                       End Projections

                                in filename.win

                                shells type : list of dictionaries which values are int

                    - ham_nkpt: k-point sampling needed for the creation of Hk from H_R.
                                In is built from  mp_grid keyword from filename.win file.

                                ham_nkpt: list of int

                    - SP:   defines whether we have spin-polarised or spin-less calculation. \n
                            Possible values:

                                * SP=0 spin-less calculations

                                * SP=1 spin-polarised calculation

                            If filename.win is present then SP=0,
                            if filename_up.win and filename_down.win are both present then SP=1

                            SP type: int

                    - SO:   defines whether or not we have spin-orbit calculations.\n
                            Possible values are:

                                * SO=0 no spin-orbit calculations

                                * SO=1 spin-orbit calculations

                            SO type: int

                            If filename.win is present and inside of it there is keyword  spinors=true
                            then SO=1, otherwise SO=0

                     While reading keywords from filename,win Fortran convention is honoured
                    (cases of  letters do not matter,'!' starts a comment).

        :param corr_shells: user defined correlated shells, in some rare cases it
                       may happen that orbitals which are usually considered non-correlated states
                       should  be treated as correlated states,
                       like for example O states in RbO2 can be treated as correlated states
                       (see .Phys. Rev. B 80, 140411 (2009) for more details).
                       This parameter should be set by user only in such non-standard situation. If this
                       parameter is not set (it has its default value None), then shells and
                       corr_shells will be constructed according to standard convention:
                       s,p states are non-correlated states
                       d,f states are correlated states
                       If user provided shells are different than None then shell and corr_shells  have
                       the same value.

        :type corr_shells: list of dictionaries
        """

        #*********************check data from the previous run **********************************
        things_to_load=["n_k", "SP", "SO", "density_required", "shells","n_shells",
                        "corr_shells", "n_corr_shells", "T", "n_orbitals", "proj_mat",
                        "hopping", "n_inequiv_shells", "corr_to_inequiv",
                        "inequiv_to_corr", "rot_mat", "energy_unit",
                        "k_dep_projection", "charge_below", "symm_op",
                        "use_rotations", "rot_mat_time_inv", "n_reps", "dim_reps"]

        self._all_read = self.__read_parameters_from_h5_file(   subgrp=self._sumk_dft,
                                                                thing_to_load=things_to_load)


        if not self._all_read:  # calculation from scratch

            # non standard shells
            if not corr_shells is None:
                for n_corr in range(len(corr_shells)):
                    self.check_shell(x=corr_shells[n_corr],
                                     t=["atom","sort", "l", "dim", "SO", "irep" ])
                self.corr_shells=corr_shells
                self.shells=deepcopy(self.corr_shells)
                for n_shell in range(len(self.shells)):
                    del self.shells[n_shell]["SO"]
                    del self.shells[n_shell]["irep"]

            self._read_win_file()
            # if not (isinstance(ham_nkpt, list) and len(ham_nkpt) == 3 and
            #     all([isinstance(ham_nkpt[i], int) and ham_nkpt[i]>0 for i in range(len(ham_nkpt))])):
            #     self.report_error("You have to define a proper k-mesh!")

            # calculate all other parameters
            self.n_corr_shells=len(self.corr_shells)
            self.n_shells=len(self.shells)

            # Determine the number of inequivalent correlated shells, has to be known for further reading...
            self.n_inequiv_corr_shells, self.corr_to_inequiv, self.inequiv_to_corr=self.det_shell_equivalence(corr_shells=self.corr_shells)

            # parameters needed by both rerun and fresh run
            self.n_k = self.ham_nkpt[0] * self.ham_nkpt[1] * self.ham_nkpt[2] #number of k-points

            self._name2SpinChannel={self._filename+"_up":0,
                                    self._filename+"_down":1,
                                    self._filename:0 }

            self._disentangled_spin={i: False for i in range(self._n_spin)}

            self._parameters={"num_zero":self._num_zero,
                              "verbosity":self._verbosity,
                              "ham_nkpt":self.ham_nkpt}

            # delete content of hdf file, since some parameters were not found
            # it is invalid and start to  write file from scratch

            if mpi.is_master_node():
                ar = HDFArchive(self._filename+".h5","w")
                del ar

            self.k_point_mesh = None
            self._R_sym=[]
            for n_corr in range(self.n_corr_shells):

                self._R_sym.append({"atom":self.corr_shells[n_corr]["atom"],
                                    "sort":self.corr_shells[n_corr]["sort"],
                                    "rot_mat":numpy.zeros((self.corr_shells[n_corr]['dim'],
                                                          self.corr_shells[n_corr]['dim']),
                                                            numpy.complex),
                                    "eig":numpy.zeros(self.corr_shells[n_corr]['dim'],dtype=numpy.float)})

                #R_sym is  data structure to store symmetry operation, it has a form of list,
                # each entry correspond to one correlated shell, entry is a dictionary with following keywords:
                #   "atom":     number of atom
                #   "sort" :    label of symmetry equivalent atom, all equivalent atoms have the same sort
                #   "rot_mat":  rotation matrix
                #   "eig":      eigenvalues which correspond to the particular correlated site

            self.__h_to_triqs()
            self._get_density_required()
            self._produce_projectors()

            self.proj_mat=mpi.bcast(self.proj_mat)
            self.n_orbitals=mpi.bcast(self.n_orbitals)
            self._dummy_projectors_used=mpi.bcast(self._dummy_projectors_used)
            self._u_matrix_full=mpi.bcast(self._u_matrix_full)
            self.total_Bloch=mpi.bcast(self.total_Bloch)

            self._produce_hopping()
            self._sumk_dft_par()

            if mpi.is_master_node():

                self._save_par_hdf(name=self._sumk_dft,
                               dictionary=self._parameters)

                self._save_par_hdf(name=self._sumk_dft,
                               dictionary=self._sumk_dft_data)

        else:  # case of rerun

            # warn if input parameters for sumk_dft have changed their values since the last run

            sumk_dft_data={ "energy_unit":1.0,
                            "k_dep_projection":0 ,
                            "SP":self.SP,
                            "SO":self.SO,
                            "charge_below":0,
                            "density_required":self.density_required,
                            "symm_op":0,
                            "n_shells":len(self.shells),
                            "shells":self.shells,
                            "n_corr_shells":len(self.corr_shells),
                            "corr_shells":self.corr_shells,
                            "use_rotations":1,
                            "rot_mat_time_inv":[0 for i in range(self.n_shells)],
                            "n_reps":-1,
                            "dim_reps":-1,
                            "T":self.T,
                            "n_orbitals":self.n_orbitals
                            }

            #critical parameters, if they change between reruns program should stop
            parameters=self._parameters.keys()
            not_crucial_par=["verbosity","num_zero"]
            for item in not_crucial_par:
                if item in parameters: parameters.remove(item)

            self._critical_par=sumk_dft_data.keys()+parameters
            self._critical_par_changed=False

            # warn if input  parameters for sumk dft have changed their values since the last rerun
            self.check_parameters_changed(dictionary=sumk_dft_data,
                                        hdf_dir=self._sumk_dft)

            # warn if input  parameters for Wannier_converter have changed their values since the last rerun
            if self._critical_par_changed: self._critical_par_changed=False #start a new search

            self.check_parameters_changed(dictionary=self._parameters,
                                        hdf_dir=self._sumk_dft)

        mpi.barrier()


    def _read_win_file(self):


        """
        Extracts parameters from win file.

        """
        reader=AsciiIO()

        one_channel = False

        try:

            with open(self._filename+".win") as input_file:
                one_channel = True
                input_file.close()

        except IOError:

            try:

                with open(self._filename+"up.win") as input_file_1, open(self._filename+"down.win") as input_file_2:
                    input_file_1.close()
                    input_file_2.close()

            except IOError:

                self.report_error("Couldn't find any of the following obligatory files (option 1) or 2) ): "+
                                  "1) "+self._filename+".win or"+
                                  "2) "+self._filename+"up.win and "+self._filename+"down.win " )

        if one_channel:

            reader.read_ASCII_fortran_file(file_to_read=self._filename+".win",default_dic=self._win_par)

            self._get_spinors()
            self.SP=0

        else:
            reader.read_ASCII_fortran_file(file_to_read=self._filename+"up.win",default_dic=self._win_par)
            self.SO=0
            self.SP=1

        # **** calculate parameters for converter based on parameters from win file
        if self.shells is None:
            self._get_shells()
        self._get_T()
        self._get_ham_nkpt()
        self._get_chemical_potential()


        # **** read extra parameters ****
        try:
            with open(self._extra_par_file) as input_file:
                extra_file=True
                input_file.close()
        except IOError:
                extra_file=False

        if extra_file:
            reader.read_ASCII_file( file_to_read=self._extra_par_file,
                                    default_dic=self._extra_par,
                                    variable_list_string_val=[])
        else:
            self.make_verbose_statement("Default values of verbosity and num_zero will be used.")

        # set all extra parameters
        try:
            self._num_zero=float(self._extra_par["num_zero"])
        except ValueError:
            self.report_error("Wrong type of parameter num_zero!")

        try:
            self._verbosity=int(self._extra_par["verbosity"])
        except ValueError:
            self.report_error("Wrong type of parameter verbosity!")

        # check if extra parameters have reasonable values
        if not  1.0>self._num_zero>0.0:
            self.report_error("Numerical zero is wrongly defined!")

        if self._verbosity != 2:
            self.make_statement("No additional output will be printed out from lattice module.")
            self._verbosity = "None"


    def _get_spinors(self):
        """
        Calculates spinors from data which was extracted from *.win file. Sets value of SO.

        """

        if not (self._win_par["spinors"].lower() == "true"  or
                self._win_par["spinors"].lower()=="false" or
                self._win_par["spinors"].lower() == "t" or
                self._win_par["spinors"].lower()=="f"):
            self.report_error("Invalid value of spinors!")

        spinors= (self._win_par["spinors"][0].upper()+self._win_par["spinors"][1:].lower())=="True"

        if spinors:
            self.SO=1
        else:
            self.SO=0


    def _get_chemical_potential(self):

        """
        Calculates initial guess for chemical potential from data which was extracted from *.win file.

        """

        try:
           self.chemical_potential=float(self._win_par["fermi_energy"])
        except ValueError:
            self.report_error("Invalid value of fermi_energy! ")


    def _get_ham_nkpt(self):
        """
        Calculates vectors with 3 elements which defines k-mesh
        from data which was extracted from *.win file.

        """

        temp_mp_grid=self._win_par["mp_grid"].split()
        self.ham_nkpt=[]

        try:
            for i in range(len(temp_mp_grid)): self.ham_nkpt.append(int(temp_mp_grid[i]))
        except ValueError:
            self.report_error("Invalid value of mp_grid!")


    def _get_shells(self):
        """

        Calculates shells and corr_shells basing on
        data which was extracted from *.win file.
        Value of "sort" keyword  is found later from HR.

        """

        lines=self._win_par["projections"].split("\n")
        lines.pop(len(lines)-1) # remove \n from the last line
        self.shells=[]
        self._atoms={}
        num_atom=-1
        if not ":" in lines[0]: lines.pop(0) # in case first line in the block is [units] remove it
        for line in lines: # each line is a different atom, but atoms can repeat themselves
            label=(line[:line.find(":")]).lower() # case of the label does not matter
            begin=line.find(":")+1  # here we add 1 because we want content after

            if ":" in line[begin:]:
                end=line[begin:].find(":")+begin
            else:
                end=len(line)

            projections=line[begin:end].split(";")

            i=0
            if not label in self._atoms:
                num_atom+=1
                self._atoms[label]={"num_atom":num_atom,"state":[]}
                res=self._get_proj_l(input_proj=projections.pop(0),atom_name=label)
                temp_shells=[{"atom":num_atom,"dim":res["dim"],"l":res["l"]}]
                for proj in projections:
                    res=self._get_proj_l(input_proj=proj, atom_name=label)
                    l=res["l"]
                    dim=res["dim"]
                    found=False
                    for i,item in enumerate(temp_shells):
                        if l == item["l"]:
                            item["dim"]+=dim
                            found = True
                            break
                    if not found: # different shell on the same atom, with different l
                        temp_shells.append({"atom":num_atom,"dim":dim,"l":l})

                self.shells.extend(temp_shells)

            elif label in self._atoms:

                local_num_atom=self._atoms[label]["num_atom"]
                for proj in projections:
                    res=self._get_proj_l(input_proj=proj,atom_name=label)
                    l=res["l"]
                    dim=res["dim"]
                    found=False
                    for i,item in enumerate(self.shells):
                        if l == item["l"] and item["atom"]==local_num_atom:
                            item["dim"]+=dim
                            found=True
                            break
                    if not found: # different shell on the same atom, with different l
                        self.shells.append({"atom":local_num_atom, "dim":dim,"l":l})

        for state in self.shells:

            if state["l"] <0: # non-correlated hybridized states
                hyb_found=False
                for entry in self._all_states:
                    if state["l"]== entry["l"] and  state["dim"]!= entry["dim"]:
                        hyb_found=True
                        break
                if not hyb_found:
                    self.report_error("Wrong structure of shells!")
            elif state["dim"]>(state["l"]*2+1):
                self.report_error("Wrong structure of shells!")


    def _get_corr_shells(self):

        """

        Calculates corr_shells from shells.

        """

        if self.shells is None: self.report_error("Shells not set yet. They are needed to calculate corr_shells!")
        self.corr_shells=[]
        for state in self.shells:
            if state["l"]==2 or state["l"]==3:
                corr_state=deepcopy(state)
                corr_state["SO"]=self.SO
                corr_state["irep"]=0
                self.corr_shells.append(corr_state)


    def _get_proj_l(self, input_proj=None, atom_name=None):
        """



        :param input_proj: Initial guess for MLWF to be analysed
        :type input_proj: str

        :param atom_name: Name of atom from *.win file for which shells are to be found
        :type atom_name: str

        :return: l, mr,dim for  the given initial guess for MLWF
        :rtype : dictionary
        """

        proj_all_entries=input_proj.replace(","," ").replace("="," ").lower().split()
        l=None
        dim=0
        indx=0
        if not isinstance(input_proj,str):
            self.report_error("Projection %s is an invalid initial projection!"%proj_all_entries)

        # case: l=foo1, mr=foo2,foo3,...
        if len(proj_all_entries)>=4:

            if "l" in proj_all_entries:
                indx=proj_all_entries.index("l")+1
                if indx==len(proj_all_entries): self.report_error("Projection %s is an "
                                                                  "invalid initial projection!"%proj_all_entries)

                # convert value l from string to int
                try:
                    l=int(proj_all_entries[indx])
                except ValueError:
                    self.report_error("Invalid value of l!")

            else:
                self.report_error("Projection %s is an invalid"
                                  " initial projection!"%proj_all_entries)

            indx+=1
            if not "mr" == proj_all_entries[indx]:

                self.report_error("Invalid initial projection! No  mr keyword found!")

            mr_list=proj_all_entries[indx+1:] # stores all mr values

            # convert string elements to int values
            try:
                for mr_indx in range(len(mr_list)):
                    mr_list[mr_indx]= int(mr_list[mr_indx])
            except ValueError:
                self.report_error("Invalid value of mr!")

            for mr_item in mr_list:

                mr_found=False

                for state in self._all_states:

                    if state["l"] == l and state["mr"] == mr_item:
                        dim+= state["dim"]
                        mr_found=True

                        # check if the same projection not repeated
                        self._is_unique_proj(atom_name=atom_name,state=state)

                        break

                if not mr_found:
                     self.report_error("Invalid value of mr!")

        # case: l=foo
        elif len(proj_all_entries)==2:
            if "l" in proj_all_entries:
                indx=proj_all_entries.index("l")+1
                if indx==len(proj_all_entries): self.report_error("Projection %s is an invalid"
                                                                  " initial projection!"%proj_all_entries)
                l=proj_all_entries[indx]
            else:
                self.report_error("Projection %s is an invalid "
                                  "initial projection!"%proj_all_entries)

            for state in self._all_states:

                if state["l"] == l and  state["mr"] is None:
                    dim = state["dim"]

                    # check if the same projection not repeated
                    self._is_unique_proj(atom_name=atom_name,state=state)

                    break

            if dim ==0:
                self.report_error("Invalid value of l!")

        # case: foo
        elif len(proj_all_entries)==1:

            for state in self._all_states:
                if state["name"]==proj_all_entries[0]:
                    l=state["l"]
                    dim=state["dim"]
                    # check if the same projection not repeated
                    self._is_unique_proj(atom_name=atom_name,state=state)
                    break

            if dim==0:
                self.report_error("Projection %s is an invalid "
                                  "initial projection!"%proj_all_entries)

        else:
             self.report_error("Projection %s is an invalid "
                               "initial projection!"%proj_all_entries)

        return  {"l":l,"dim":dim}


    def _get_T(self):
        pass


    def _get_required_density(self):
        pass


    def __h_to_triqs(self):
        """
        Reads  HR file calculated by wannier90  and calculates Hk and symmetry operations.
        Detailed steps below:

            * Reads filename_hr.dat

            * Constructs symmetry operators

            * Constructs Hk

            * Updates parameters dictionary

        """

        LOCAL_VARIABLES = ["ham_r", "nrpt"]

        GLOBAL_VARIABLES = ["total_MLWF", "FULL_H_R",
                            "Vector_R_degeneracy",
                            "Vector_R"]

        for it in LOCAL_VARIABLES:
            exec "%s = 0" % it

        for it in GLOBAL_VARIABLES:
            exec "self.%s=0" % it

        if self._n_spin == 2:
            results=self.__read_spin_H_R

        else:
            results=self._read_H_R_file(filename=self._filename)

        for it in GLOBAL_VARIABLES:
            exec "self.%s = results[it]"%it

        for it in LOCAL_VARIABLES:
            exec "%s =  results[it]"%it


        # 2) construct symmetry operators
        # local_shells=[ {} for n in range(self.n_shells)]
        # total_orb=0
        # for ish in range(self.n_shells):
        #
        #     n_orb=self.shells[ish]["dim"]
        #     temp_eig,temp_rot_mat=numpy.linalg.eigh(ham_r[total_orb:total_orb+n_orb,total_orb:total_orb+n_orb])
        #
        #     #sort eigenvectors and eigenvalues
        #     temp_indx = temp_eig.argsort()
        #     local_shells[ish]["eig"]=temp_eig[temp_indx]
        #     local_shells[ish]["rot_mat"]=temp_rot_mat[:,temp_indx]
        #
        #     #check if shell is a correlated shell if so write eigenvectors to rotation which corresponds to it
        #     for icrsh in range(self.n_corr_shells):
        #
        #         if self.compare_shells(shell=self.shells[ish],
        #                                corr_shell=self.corr_shells[icrsh]):
        #
        #             self._R_sym[icrsh]["eig"]= local_shells[ish]["eig"]
        #             self._R_sym[icrsh]["rot_mat"]=local_shells[ish]["rot_mat"]
        #             break
        #
        #     #in case of spin-polarised calculation we run only through "up"
        #     # block of ham_r, it is ok because we look for spacial symmetry here
        #     total_orb+=n_orb
        #
        # #check if initial projections from *win file are consistent with shells provided in the input file
        # for ish1 in range(self.n_shells):
        #     for ish2 in range(self.n_shells):
        #
        #         #equivalent shell found
        #         if self.is_shell(self.shells[ish1], self.shells[ish2]):
        #
        #             if  self._num_zero < numpy.min(numpy.abs(local_shells[ish1]["eig"]-local_shells[ish2]["eig"])):
        #                  self.report_error("Wrong block structure of input Hamiltonian,"
        #                                    " correct it please (maybe num_zero parameter is set too low?)!")
        #             break
        #
        # del local_shells

        if self.total_MLWF!=sum([ sh['dim'] for sh in self.shells ]):

            self.report_error("Wrong block structure of input Hamiltonian, correct it please!")

        self.rot_mat = [numpy.zeros((self.corr_shells[icrsh]['dim'],
                                self.corr_shells[icrsh]['dim']),
                               numpy.complex) for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            self.rot_mat[icrsh]=numpy.dot(self._R_sym[icrsh ]["rot_mat"],
            self._R_sym[self.corr_to_inequiv[icrsh]]["rot_mat"].conjugate().transpose())

        #3) Constructs Hk
        self.Hk = [[numpy.zeros((self.total_MLWF, self.total_MLWF), dtype=complex)
                              for isp in range(self._n_spin)] for ikpt in range(self.n_k)]

        self.__make_k_point_mesh()
        imag = 1j
        twopi = 2 * numpy.pi

        total_n_kpt = self.ham_nkpt[0] * self.ham_nkpt[1] * self.ham_nkpt[2]

        for ikpt in range(total_n_kpt):
            for n_s in range(self._n_spin):
                indices_nrpt = numpy.array(range(nrpt))
                #parallelization over nrpt
                for irpt in mpi.slice_array(indices_nrpt):
                    rdotk = twopi * numpy.dot(self.k_point_mesh[ikpt], self.Vector_R[irpt])
                    factor = (math.cos(rdotk) + imag * math.sin(rdotk)) / float(self.Vector_R_degeneracy[irpt])

                    self.Hk[ikpt][n_s][:, :] += factor * self.FULL_H_R[irpt][
                    self.total_MLWF * n_s:
                    self.total_MLWF * (n_s + 1),
                    self.total_MLWF * n_s:
                    self.total_MLWF * (n_s + 1)]

                self.Hk[ikpt][n_s][:, :] = mpi.all_reduce(mpi.world,self.Hk[ikpt][n_s][:, :],lambda x_ham, y: x_ham + y)

                mpi.barrier()

        #4) update parameters dictionary

        Variables = {"k_point_mesh": self.k_point_mesh, "FULL_H_R": self.FULL_H_R,
                     "Vector_R": self.Vector_R,
                     "Vector_R_degeneracy": self.Vector_R_degeneracy,
                     "Hamiltonian": self.Hk,
                     "R_sym": self._R_sym}

        self._parameters.update(Variables)


    def _read_H_R_file(self,filename=None):

        """
        Reads H_R from the text file.
        :param filename: name of H_R file to read
        :type filename: str
        :return: dictionary with data which was read form filename_hr.dat

        """

        local_variable={"nrpt":None,"total_MLWF":None,
                "Vector_R_degeneracy":None,
                "Vector_R":None,  "FULL_H_R":None, "ham_r":None}

        if mpi.is_master_node():
            try:
                with open(filename + "_hr.dat", "rb") as hr_txt_file:
                    hr_file = cStringIO.StringIO(hr_txt_file.read())  # writes file to memory to speed up reading
                    hr_txt_file.close()
                    dimensions = 3
                    hr_file.readline()  # first line == date
                    total_MLWF = int(hr_file.readline())
                    # reads number of Wannier functions per spin
                    nrpt = int(hr_file.readline())
                    Vector_R_degeneracy = numpy.zeros(nrpt, dtype=int)
                    Vector_R = numpy.zeros((nrpt, dimensions), dtype=int)

                    FULL_H_R = [numpy.zeros((total_MLWF,
                                            total_MLWF),
                                            dtype=complex)
                                      for n in range(nrpt)]

                    ham_r = numpy.zeros((total_MLWF,
                                        total_MLWF),
                                        dtype=complex)

                    counter = 0
                    while counter < nrpt:
                        for x in hr_file.readline().split():
                            Vector_R_degeneracy[counter] = x
                            counter += 1

                    for irpt in range(nrpt):
                        for jj in range(total_MLWF):
                            for ii in range(total_MLWF):
                                line = [float(x) for x in hr_file.readline().split()]

                                if ii == 0 and jj == 0:
                                    Vector_R[irpt] = numpy.array([int(line[0]), int(line[1]), int(line[2])])

                                indx_i=int(line[3])-1 #we count from zero
                                indx_j=int(line[4])-1
                                if ii!= indx_i or jj!=indx_j:
                                    self.report_error("Inconsistent indices for H_R: [%s!=%s,%s!=%s]!"%(ii,indx_i,jj,indx_j))

                                FULL_H_R[irpt][ii, jj] = complex(float(line[5]), float(line[6]))
                                if int(line[0]) == 0 and int(line[1]) == 0 and int(line[2]) == 0:
                                    ham_r[ii, jj] = complex(float(line[5]), float(line[6]))  #zeroth unit cell found

                        #check if  FULL_H_R[irpt] is real, for well localized MLWF HR should be real
                        if self._num_zero<numpy.abs((FULL_H_R[irpt].imag.max()).max()):
                            self.report_error("HR in MLWF has large complex components!")

                    #check if ham_r is symmetric
                    if not numpy.allclose(ham_r.transpose(),ham_r,atol=self._num_zero):
                        self.report_error("Your Hamiltonian is not symmetric!")

                    hr_file.close()

                local_variable.update({"nrpt":nrpt,"total_MLWF":total_MLWF,
                                       "Vector_R_degeneracy":Vector_R_degeneracy,
                                       "Vector_R":Vector_R,  "FULL_H_R":FULL_H_R,
                                       "ham_r":ham_r})

            except IOError:
                self.report_error("Opening file %s_hr.dat failed!" %filename)

        local_variable=mpi.bcast(local_variable)
        mpi.barrier()

        return local_variable


    def __read_parameters_from_h5_file(self, subgrp=None, thing_to_load=None,just_check=True):
        """
        Reads data from parameters directory in self._filename.h5,
        necessary for object of lattice class to function correctly.

        :param subgrp: 			Folder in  self._filename.h5 where data is stored,
        :type subgrp: 			str

        :param thing_to_load:  	list of items in subgrp folder needed to do rerun.
        :type thing_to_load: 	list of str

        :param just_check:      if set to true then only checks if items are in the
                                directory  subgrp, otherwise it will be loaded
        :type                   bool

        :return:                True if all items from the list found otherwise False
        :rtype: bool

        """
        if not just_check:
            for it in thing_to_load:
                exec "self.%s = 0" % it
        found_all = True
        try:
            if mpi.is_master_node():
                if isfile(self._filename + ".h5"):
                    ar = HDFArchive(self._filename + ".h5", "a")
                    if subgrp not in ar:

                        self.report_warning(
                            "%s not found in %s.h5 file. %s.h5 will be created from scratch. " % (subgrp,
                                                                                                  self._filename,                                                                          self._filename))
                        found_all = False

                    else:
                        for it in thing_to_load:

                            if it in ar[subgrp]:
                                if not just_check:
                                    exec "self.%s = ar['%s'][it]" % (it, subgrp)
                            else:
                                if not just_check:
                                    self.report_warning("Loading %s failed!" % it)
                                else:
                                    self.report_warning("%s not found!" % it)

                                self.report_warning("%s.h5 will be created from scratch." % self._filename)
                                found_all = False
                                break

                    del ar
                else:
                    found_all = False
            found_all = mpi.bcast(found_all)

        except IOError:
            self.report_warning("Opening file %s.h5 failed!" % self._filename)
            found_all = False

        if found_all and not just_check:

            for it in thing_to_load:
                exec "self.%s = bcast(self.%s)" % (it, it)

        return found_all


    @property
    def __read_spin_H_R(self):
        """In case of spinpolarised calculations two files  filename+[up, down]'_hr.dat'
           are obtained from wannier90.x, each corresponds to the different
           spin channel. This method produces combined filename_hr.dat which
           is then used in DMFT calculations."""

        # initialize it so that result variable is visible to all threads
        result={"nrpt":None,"total_MLWF":None,
                "Vector_R_degeneracy":None,
                "Vector_R":None,  "FULL_H_R":None, "ham_r":None}

        # reads spin channel up
        results=self._read_H_R_file(filename=self._filename + "_up")
        nrpt_up=results["nrpt"]
        total_MLWF_up=results["total_MLWF"]
        Vector_R_degeneracy_up=results["Vector_R_degeneracy"]
        Vector_R_up=results["Vector_R"]
        FULL_H_R_up=results["FULL_H_R"]
        ham_r_up=results["ham_r"]

        # reads spin channel down
        results=self._read_H_R_file(filename=self._filename + "_down")
        nrpt_down=results["nrpt"]
        total_MLWF_down=results["total_MLWF"]
        Vector_R_degeneracy_down=results["Vector_R_degeneracy"]
        Vector_R_down=results["Vector_R"]
        FULL_H_R_down=results["FULL_H_R"]
        ham_r_down=results["ham_r"]

        if nrpt_up != nrpt_down:
            self.report_error(self._filename + "down_hr.dat has a different grid " +
                            "in real space  than " + self._filename + "up_hr.dat!")

        if total_MLWF_up != total_MLWF_down:
            self.report_error(self._filename + "down_hr.dat has a different number " +
                            "of Wannier functions than " + self._filename + "up_hr.dat!")

        if not (Vector_R_degeneracy_up.shape==Vector_R_degeneracy_down.shape  and
                numpy.allclose(Vector_R_degeneracy_up,Vector_R_degeneracy_down)):

            self.report_error("Inconsistent degeneracies of R-vectors in channel up and down!")

        if not (Vector_R_up.shape==Vector_R_down.shape and
                    numpy.allclose(Vector_R_up,Vector_R_down)):

            self.report_error("Inconsistent R vectors for channel up and down!")


        # write combined hr.dat
        combined_num_orb = self._n_spin * total_MLWF_down
        FULL_H_R = [numpy.zeros((    combined_num_orb,
                                           combined_num_orb),
                                      dtype=complex)
                          for n in range(nrpt_down)]

        ham_r = numpy.zeros((   combined_num_orb,
                                combined_num_orb),
                                dtype=complex)

        for n in range(nrpt_down):
            FULL_H_R[n][:total_MLWF_down, :total_MLWF_down] = FULL_H_R_up[n]
            FULL_H_R[n][total_MLWF_down:combined_num_orb, total_MLWF_down:combined_num_orb] = FULL_H_R_down[n]

        ham_r[:total_MLWF_down, :total_MLWF_down]=ham_r_up
        ham_r[total_MLWF_down:combined_num_orb, total_MLWF_down:combined_num_orb]=ham_r_down

        result.update({"nrpt":nrpt_down, "total_MLWF":total_MLWF_down,
                 "Vector_R_degeneracy":Vector_R_degeneracy_down,
                 "Vector_R":Vector_R_down,  "FULL_H_R":FULL_H_R, "ham_r":ham_r})

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
        self.k_point_mesh = numpy.zeros((self.n_k, dimensions), dtype=float)
        n_kpt = 0
        for i1 in range(i1min, i1max + 1):
            for i2 in range(i2min, i2max + 1):
                for i3 in range(i3min, i3max + 1):
                    self.k_point_mesh[n_kpt, :] = [float(i1) / float(self.ham_nkpt[0]),
                                                    float(i2) / float(self.ham_nkpt[1]),
                                                    float(i3) / float(self.ham_nkpt[2])]
                    n_kpt += 1


    # def is_shell(self,shell_1st=None,shell_2nd=None):
    #     """
    #
    #     Checks if shell_1st is equal to  shell_2nd
    #
    #     :param shell_1st: first shell to compare
    #     :type shell_1st: int
    #
    #     :param shell_2nd:  second shell to compare
    #     :type shell_2nd: int
    #
    #     :return: True if shell_1st is shell_2nd otherwise False
    #
    #     """
    #
    #     if not self.check_shell(shell_1st,t=self.__class__.shells_keywords):
    #         self.report_error("Shell was expected!")
    #     if not self.check_shell(shell_2nd,t=self.__class__.shells_keywords):
    #         self.report_error("Shell was expected!")
    #
    #     return cmp(shell_1st,shell_2nd)==0


    def _is_unique_proj(self,atom_name=None, state=None):

        """
        Checks if projections defined by state input parameters
        were not defined before, if it was defined then program
        prints info about repetition and terminates.

        :param state: dictionary with the projection
        :type state: dict
        """

        if not state in self._atoms[atom_name]["state"]:
            self._atoms[atom_name]["state"].append(state)
        else:
            self.report_error("Repetition of the projection for %s"
                                              " for atom labeled by: %s"%(state,atom_name))


    def compare_shells(self,shell=None, corr_shell=None):
        """
        Checks if shell corresponds to corr_shell

        :param shell: shell to compare
        :type shell: dict of int
        :param corr_shell:  correlated shell to compare
        :type corr_shell: dict of int
        :return: True if shell corresponds to corr_shell otherwise False
        :rtype: bool
        """

        if not self.check_shell(corr_shell, t=self.__class__.corr_shells_keywords):
            self.report_error("Correlated shell was expected!")
        if not self.check_shell(shell,t=self.__class__.shells_keywords):
            self.report_error("Shell was expected!")

        return (shell["atom"]==corr_shell["atom"] and
                shell["sort"]==corr_shell["sort"] and
                shell["l"]==corr_shell["l"] and
                shell["dim"]==corr_shell["dim"])


    def eval_offset(self,n_corr=None):
        """

        Calculates position of block in matrix which corresponds
        to the n_corr-th correlated shell, position is defined by
        offset: beginning of block ('upper left corner' of block)
        and size of block (its 'width')

        :param n_corr: number of correlated shell
        :type n_corr: int
        :return:  * n_orb: number of states for the n_corr-th correlated shell

                  * offset: position of block in matrix (for example U matrix )
                    which corresponds to the n_corr-th correlated shell
        :rtype: tuple of ints
        """

        if not self.check_n_corr(n_corr=n_corr):
            self.report_error("Invalid number of correlated shell!")

        # calculate the offset:
        offset = 0
        n_orb=0
        for ish in range(self.n_shells):

            if self.compare_shells(shell=self.shells[ish],
                                   corr_shell=self.corr_shells[n_corr]):

                n_orb = self.corr_shells[n_corr]['dim']

                break

            else:

                offset += self.shells[ish]['dim']

        if n_orb==0:

                self.report_error("Inconsistency between shells and corr_shells!")

        return n_orb,offset


    def _set_U_matrices_None(self):
        """
        to be used in case no valid *win or *.chk file were find.
        Sets the following objects related to projections to None:

            * u_matrix

            * u_matrix_full

            * u_matrix_opt

            * n_orbitals

        """
        self._u_matrix=None
        self._u_matrix_full=None
        self._u_matrix_opt=None
        self.n_orbitals=None


    def _extract_fortran_data(self,filename=None):

        """
        Extract U matrix, from Wannier90 output for one particular spin channel.
        In case there is a disentanglement extracts also U_matrix_opt and ndimwin
        (ndimwin is how number of bands are called in Wannier90, in wannier converter
        ndimwin is called n_orbitals in order  to be consistent with names from sumkdft).

        :param filename:    Base name of file  with data
                            from Wannier90  for one particular spin channel
        :type filename: str
        """
        #read_chkpt.set_seedname(filename)

        err_msg="Dummy projectors (identity matrices) will be used in the calculation."
        " U_matrix, U_matrix_opt,ndimwin, U_full will be set to None"

        try:
            #check if we can open file
            f = open(filename+".win","rb")
            f.close()
            #read_chkpt.param_read()
        except IOError:
           self.report_warning("Opening file %s.win failed. "%filename+err_msg)
           self._produce_dummy_projectors()
           return

        try:
            #check if we can open file
            f = open(filename+".chk","rb")
            f.close()
            #read_chkpt.param_read_chkpt()
        except IOError:
            self.report_warning("Opening file %s.chk failed. "%filename+err_msg)
            self._produce_dummy_projectors()
            return

        # if self.n_k != read_chkpt.get_num_kpts():
        #     self.report_error("Different number of input k-points and number of kpoints in %s.win! "
        #                       %filename)
        # elif self.total_MLWF  != read_chkpt.get_num_wann():
        #     self.report_error("Different number of Wannier orbitals in %s_hr.dat and in %s.win! "
        #                       %(filename,filename))

        #Prints out summary of what was found in filename.win and filename.chk files.
        # if self._verbosity==2:
        #     read_chkpt.summary()

        #disentanglement=self.__class__._fortran_boolean[read_chkpt.get_have_disentangled().strip()]

        # if filename in self._name2SpinChannel:
        #     self._disentangled_spin[self._name2SpinChannel[filename]]=disentanglement
        # else:
        #     self.report_error("Invalid name of file!")

        #Writes transformation matrix between Bloch states and smooth  pseudo- Bloch states to the self._u_matrix_opt.
        # if disentanglement:
        #
        #     #read_chkpt.write_ndimwin()  #writes n_orbitals to the text file
        #     if self.n_orbitals is None:
        #         self.n_orbitals = numpy.zeros((self.n_k,self._n_spin),numpy.int)
        #     self._read_1D_matrix(   datafile="ndimwin_"+filename+".txt",
        #                             ind1=self.n_k,
        #                             ind2=self._name2SpinChannel[filename],
        #                             matrix=self.n_orbitals,
        #                             type_el="int")
        #
        #     num_bands=self.n_orbitals.max().max()
        #     #read_chkpt.write_u_matrix_opt()
        #     if self._u_matrix_opt is None:
        #
        #         self._u_matrix_opt=numpy.zeros((num_bands,
        #                                 self.total_MLWF,
        #                                 self._n_spin,
        #                                 self.n_k),numpy.complex)
        #
        #
        #     self._read_3D_matrix(datafile="U_matrix_opt_"+filename+".txt",
        #                          ind1=num_bands,
        #                          ind2=self.total_MLWF,
        #                          ind4=self.n_k,
        #                          ind3=self._name2SpinChannel[filename],
        #                          matrix=self._u_matrix_opt)
        #
        # else:
        #     if self.n_orbitals is None:
        #         self.n_orbitals = numpy.zeros((self.n_k,self._n_spin),numpy.int)
        #
        #     for kpt in range(self.n_k):
        #         self.n_orbitals[kpt][self._name2SpinChannel[filename]]=self.total_MLWF
        #
        #
        # read_chkpt.write_u_matrix()
        if self._u_matrix is None:
            self._u_matrix=numpy.zeros((self.total_MLWF,
                                        self.total_MLWF,
                                        self._n_spin,
                                        self.n_k),numpy.complex)

        # self._read_3D_matrix(datafile="U_matrix_"+filename+".txt",
        #                     ind1=self.total_MLWF,
        #                     ind2=self.total_MLWF,
        #                     ind4=self.n_k,
        #                     ind3=self._name2SpinChannel[filename],
        #                     matrix=self._u_matrix)



        #clean up allocated fortran arrays
        #read_chkpt.dealloc()

        if self._verbosity!=2:
            try:
                remove("U_matrix_"+filename+".txt")
            except OSError:
                self.report_warning("No file"+"U_matrix_"+filename+".txt "+"found!")

            # if disentanglement:
            #     try:
            #         remove("U_matrix_opt_"+filename+".txt")
            #     except OSError:
            #         self.report_warning("No file"+"U_matrix_opt_"+filename+".txt "+"found!")
            #
            #     try:
            #         remove("ndimwin_"+filename+".txt")
            #     except OSError:
            #         self.report_warning("No file"+"ndimwin_"+filename+".txt "+"found!")


    def _produce_full_U_matrix(self):
        """
        Produces full rotation matrix from Bloch states to MLWF

        """
        #  we have 2d array here, first max returns array with maximum
        #  values from each sub-array second max returns
        #  maximum value of the whole n_orbitals
        self.total_Bloch=self.n_orbitals.max().max()

        self._u_matrix_full=numpy.zeros((self.n_k,
                                         self._n_spin,
                                         self.total_MLWF,
                                         self.total_Bloch),
                                         numpy.complex)

        temp_array=numpy.zeros((self.total_Bloch,
                                self.total_MLWF),
                                numpy.complex)

        if self._u_matrix_opt is None and self._u_matrix is None:
            self.report_warning("U matrices are not set!")
            self._u_matrix_full=None

        for k in range(self.n_k):
            for spin_bloc in range(self._n_spin):
                for i in range(self.total_Bloch):
                    for j in range(self.total_MLWF):
                        temp_array[i,j]=0.0
                        if self._disentangled_spin[spin_bloc]:

                            for orb in range(self.total_MLWF):

                                temp_array[i,j]+= self._u_matrix_opt[i,orb,spin_bloc,k] * self._u_matrix[orb,j,spin_bloc,k]

                        else:

                                temp_array[i,j]=self._u_matrix[i,j,spin_bloc,k]


                # convention in Wannier90: O' = U^{\dagger} OU
                # where O in Bloch basis and O' in MLWF basis
                #
                # convention in wannier converter
                # O' = U O U ^{\dagger}
                # where O in Bloch basis and O' in MLWF basis
                #
                # we have to transform
                # U_matrix_full -> U_matrix_full^{\dagger}
                # for each spin channel and k-point

                self._u_matrix_full[k,spin_bloc,:,:]=temp_array.conjugate().transpose()


    def _produce_projectors(self):
        """
        In case correlated and non correlated MLWF in the system dummy projectors will be built,
        otherwise projections from Bloch to correlated space are built.

        Produces projectors required by sumk_dft, in case *.chk file(s) is/are present
        otherwise dummy projectors in the form of identity matrices are produced.
        Projectors are produced for all correlated shells, also those which are symmetry equivalent.
        Projectors have a form of rotation from Bloch space to the correlated subspace.

        """
        self.proj_mat=None
        self._set_U_matrices_None()
        if mpi.is_master_node():

            # check if both  correlated and non correlated MLWF
            # are in the system, if so built dummy projections
            if len(self.corr_shells)<len(self.shells):
                self.make_statement("Correlated and non correlated"
                                    " states in the system found, dummy"
                                    " projections will be built.")
                self._produce_dummy_projectors()
                return

            #case of system with only correlated MLWF
            if self._n_spin==1:

                self._extract_fortran_data(filename=self._filename)

                #check if self._proj_mat is not already a dummy projectors
                if not self.proj_mat is None:

                    return

            elif self._n_spin==2:

                #up channel
                self._extract_fortran_data(filename=self._filename+"_up")

                #check if self._proj_mat is not already a dummy projectors
                if not self.proj_mat is None:

                    return

                else:

                    self._extract_fortran_data(filename=self._filename+"_down")

                #check if self._proj_mat is not already a dummy projectors
                if not self.proj_mat is None:

                    return

            else:
                 self.report_error("Invalid number of spin blocs!")

            self._produce_full_U_matrix()
            #Initialise the projectors:
            self.proj_mat = numpy.zeros((  self.n_k,
                                        self._n_spin,
                                        self.n_corr_shells,
                                        max([crsh['dim'] for crsh in self.corr_shells]),
                                        self.total_Bloch),numpy.complex)

            for icrsh in range(self.n_corr_shells):
                n_orb,offset=self.eval_offset(n_corr=icrsh)
                for ik in range(self.n_k):
                    for isp in range(self._n_spin):
                        self.proj_mat[ik,isp,icrsh,0:n_orb,:] = self._u_matrix_full[ik, isp,  offset:offset+n_orb,:]




    def _produce_dummy_projectors(self):


        """

        In case there is no valid *win, or *chk files or non-correlated MLWF
        in the system, method will clean up variables and produce dummy projections.
        Dummy projectors have a from of identity matrices and are built only
        for correlated shells. In case dummy projectors are used, hopping
        integrals will have a form of Hk which is Fourier
        transformed from HR obtained by wannier90.

        """
        self._set_U_matrices_None()
        #read_chkpt.dealloc()

        # for dummy case number of Bloch states is equal to number of MLWF
        self.total_Bloch=self.total_MLWF
        self.n_orbitals = numpy.ones((self.n_k,self._n_spin),numpy.int) * self.total_Bloch

        self._produce_projectors_dummy_core()
        self._dummy_projectors_used=True


    def _produce_projectors_dummy_core(self):
        """"

        Produces dummy projectors (identity matrices) required by sumk_dft in case chk file is not present.

        """

        # Initialise the projectors:
        self.proj_mat = numpy.zeros((  self.n_k,
                                        self._n_spin,
                                        self.n_corr_shells,
                                        max([crsh['dim'] for crsh in self.corr_shells]),
                                        self.total_Bloch),numpy.complex)

        for icrsh in range(self.n_corr_shells):
            n_orb,offset=self.eval_offset(n_corr=icrsh)
            for ik in range(self.n_k):
                for isp in range(self._n_spin):
                    self.proj_mat[ik,isp,icrsh,0:n_orb,offset:offset+n_orb] = numpy.identity(n_orb)


    def _produce_hopping(self):
        """

        Produces hopping integrals (rotated from Bloch space to MLWF) (to check if correct).

        """

        self.hopping=numpy.zeros((  self.n_k,
                                    self._n_spin,
                                    self.total_Bloch,
                                    self.total_Bloch),numpy.complex)


        if self._dummy_projectors_used:

            for k in range(self.n_k):
                for s in range(self._n_spin):

                        # transformation from list to numpy.ndarray
                        self.hopping[k,s]=self.Hk[k][s][:, :]

        else:

            for k in range(self.n_k):
                for s in range(self._n_spin):

                        #rotate from MLWF to Bloch to get hopping
                        self.hopping[k,s]=numpy.dot(self._u_matrix_full[k,s].conjugate().transpose(),
                                                     numpy.dot(self.Hk[k][s],
                                                               self._u_matrix_full[k,s]))


    def _sumk_dft_par(self):
        """
        Prepares input parameters for sumk_dft in the form of python dictionary.
        **Parameters**::

        -  energy_unit (float): the unit of energy used in the calculations, it is fixed to energy_unit=1.0 (assuming eV units)
        -  n_k (int): number of k-points used in the summation, this is read from Hr file produced by wannier90
        -  k_dep_projection (int): whether or not dimension of  projection operator depends on the k-point:

            * 0 dimension of projections do not depend on k-points

            * 1 dimension of projections  depend on k-points

            *This parameter is fixed to  k_dep_projection=0 for Wannier90Converter
             (projections are constructed from rotation  between pseudo-Bloch states and MLWF).

        - SP (int):	defines whether we have spin-polarised or spin-less calculation. \n
                Possible values:

                * SP=0 spin-less calculations

                * SP=1 spinpolarised calculation

        - SO (int): defines whether or not we have spin-orbit calculations.\n
            Possible values are:

                * SO=0   no spin-orbit calculations

                * SO=1   spin-orbit calculations

        - charge_below (float): the number of electrons in the unit cell below
          energy window, *it is set to 0 in Wannier converter*

        - density_required (float): required electron density, used in
          search for the chemical potential. This is the total number
          of electrons described in Hamiltonian expressed in MLWF basis
          (correlated and non-correlated electrons).

        - symm_op (int): defines if symmetry operation are used fot the BZ sums\n. Possible values are:

                 *   symm_op=0 all k-points explicitly included in calculations

                 *   symm_op=1 symmetry inequivalent points used, and rest k-points
                     reproduced by means of the symmetry operations

          *This parameter is fixed to  symm_op=0 for Wannier90Converter (No symmetry groups for the k-sum)*.

        - n_shells (int): number of all sites in the unit cell

        - shells (list of dictionaries with values are  int): first dimension is n_shells, second dimension is 4,
             [{"atom":index, "sort":sort, "l":l, "dim":dim },{...}]. Keywords in each entry are as follows: \n

             * index: number of the atom

             * sort: label which marks symmetry equivalent atoms

             * l: angular momentum quantum number

             * dim: number of orbitals for the particular shell

        - n_corr_shells (int): number of all correlated sites in the unit cell

        - corr_shells (list of dictionaries which values are int): first dimension is n_corr_shells,
          second dimension is 6, [{"atom":index, "sort":sort, "l":l, "dim":dim ,"SO":SO, "irep":irep].
          Keywords in each entry are as follows:\n

             * index: number of the atom

             * sort: label which marks symmetry equivalent atoms

             * l: angular momentum quantum number

             * dim: number of orbitals for the particular shell

             * SO: spin orbit coupling, if it is enabled it is 1 otherwise it is 0

             * irep: dummy parameter set to 0

        - use_rotations (int): if local of global coordinates are used,
          *This parameter is fixed to use_rotation=1 for Wannier90Converter*

        - rot_mat (numpy.array.complex): if use_rotation=1, then this contains a list of the rotation matrices.
          *This parameter is enabled for Wannier90Converter*

        - rot_mat_time_inv (list of int): this is used only if  use_rotations=1
          *This parameter is disabled for Wannier90Converter*

        - n_reps (int): Number of irreducible representations of the correlated shells,
            for example for t2g/eg splitting it wil be 2
            *This parameter is not used  by  Wannier90Converter*

        - dim_reps (list of int): Dimension of representation (for example for t2g/eg [2,3])
          *This parameter is not used  by  Wannier90Converter*

        - T (list of numpy.array.complex): Each element in the list correspond
        to the particular inequivalent correlated shell. Each element in the list is a transformation matrix
        from the spherical harmonics to MLWF basis

        -n_orbitals(numpy.array.int, dimension n_k, SP+1-SO,n_corr_shells,max(corr_shell dimension)):
         total number of orbitals for each k-point. It is k-independent in Wannier converter, it has
         a form of vector whose each element is equal to total number of states included in
         Hamiltonian expressed in MLWF.

        -proj_mat(numpy.array.complex, dimension [n_k, SP+1-SO,n_corr_shells,max(corr_shell dimension)]):
         U matrix which transforms pseudo-Bloch states to correlated MLWF.

        -bz_weights (numpy.array.float): one dimensional array which
        contains weights of the k-points for the summation.
        Since no symmetry of BZ is used all elements of this
        list have the same value: 1/n_k

        -hopping(numpy.array.complex, dimension [n_k, SP+1-SO, Max(n_orbitals), Max(n_orbitals)]):
        non-interacting H_k at each k-point.
        """

        bz_weights = numpy.ones(self.n_k,numpy.float)/ float(self.n_k)

        self._sumk_dft_data = { "energy_unit":1.0,
                                "n_k":self.n_k ,
                                "k_dep_projection":0,
                                "SP":self.SP,
                                "SO":self.SO,
                                "charge_below":0,
                                "density_required":self.density_required,
                                "symm_op":0,
                                "n_shells":self.n_shells,
                                "shells":self.shells,
                                "n_corr_shells":self.n_corr_shells,
                                "corr_shells":self.corr_shells,
                                "use_rotations":1,
                                "rot_mat":self.rot_mat,
                                "rot_mat_time_inv":[0 for i in range(self.n_shells)],
                                "n_reps":-1, # not used
                                "dim_reps":-1, # not used
                                "T":self.T,
                                "n_orbitals":self.n_orbitals,
                                "proj_mat":self.proj_mat,
                                "bz_weights":bz_weights,
                                "hopping":self.hopping,
                                "n_inequiv_shells":self.n_inequiv_corr_shells,
                                "corr_to_inequiv":self.corr_to_inequiv,
                                "inequiv_to_corr":self.inequiv_to_corr,
                                "chemical_potential":self.chemical_potential}


    #getters
    def R_sym(self,n_corr):

        """
        :param n_corr: number of correlated shell, indexes of both symmetry equivalent and
                        symmetry inequivalent correlated sites can be given
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
    def U_matrix(self,value=None):
        self.report_error("Value of U_matrix cannot be overwritten!")


    @property
    def U_matrix_opt(self):
        """

        :return:    In case there is disentanglement rotation from
                    Bloch states to smooth pseudo-Bloch states otherwise None.

        """
        return self._u_matrix_opt


    @U_matrix_opt.setter
    def U_matrix_opt(self,value=None):

         self.report_error("Value of U_opt matrix cannot be overwritten!")


    @property
    def U_matrix_full(self):
        """

        :return: Whole rotation matrix from Bloch states to MLWF.
        If dummy projectors are used it is set to None.

        """

        return self._u_matrix_full


    @U_matrix_full.setter
    def U_matrix_full(self,value=None):
        self.report_error("Value of U_full matrix cannot be overwritten!")


    @property
    def sumk_dft_input_folder(self):
        """

            :return: Folder from hdf file in which sumk_dft data is stored
            :rtype : str

        """
        return self._sumk_dft


    @sumk_dft_input_folder.setter
    def sumk_dft_input_folder(self,value=None):
        """
        Prevents user from overwriting sumk_dft_input_folder
        :param value: new potential value
        :type value: str
        """
        self.report_error("Attribute sumk_dft_input_folder cannot be modified by user!")


    def get_hopping(self,n_k,n_spin_bloc):
        """

        :param n_k: n_k-th k-point
        :type n_k: int

        :param n_spin_bloc: number of spin-bloc  (in case of spinless calculation we have only one spin-block
                            and valid value is  0, in case of spinpolarised polarised we have two blocs
                            and valid values are 0,1)
        :type n_spin_bloc: int

        :return: If valid input parameters provided returns hopping integrals
                 in the following format for the particular k-point and spin-bloc.
                 If input parameters are invalid returns None
        :rtype :numpy.array.complex, dimension [ Max(n_orbitals), Max(n_orbitals)]

        """
        if self.check_n_k(n_k=n_k) and self.check_n_spin_bloc(n_spin_bloc=n_spin_bloc):
            return self.hopping[n_k,n_spin_bloc,:,:]
        else:
            return None


    @property
    def n_inequivalent_corr_shells(self):
        return self.n_inequiv_corr_shells


    @n_inequivalent_corr_shells.setter
    def n_inequivalent_corr_shells(self,value=None):
        self.report_error("Attribute n_inequivalent_corr_shells cannot be modified by user!")


    def get_projector(self, n_k, n_spin_bloc, n_corr_shell):
        """
        :param n_k: n-th k-point
        :type n_k: int

        :param n_spin_bloc: number of spin-bloc  (in case of spinless calculation we have only one spin-block
                            and valid value is  0, in case of spinpolarised polarised we have two blocs
                            and valid values are 0,1)
        :type n_spin_bloc: int

        :param n_corr_shell: number of correlated shell (index  for inequivalent or equivalent correlated shell)
        :type n_corr_shell: int

        :return: if valid input parameters provided returns projector array otherwise return None
        :rtype: numpy.array.complex, dimension [Max(corr_shell dimensions), Max(n_orbitals)]]

        """

        if (self.check_n_k(n_k=n_k) and
            self.check_n_spin_bloc(n_spin_bloc=n_spin_bloc) and
            self.check_inequivalent_corr(n_corr=n_corr_shell)):

            return  self.proj_mat[ n_k, n_spin_bloc, n_corr_shell,:,:]

        else:

            return None


    @property
    def all_read(self):
        """


        :return: True if all items necessary for Wannier90Converter
        object to function properly were loaded otherwise False
        :rtype : bool

        """
        return self._all_read

    @all_read.setter
    def all_read(self,value=None):
       self.report_error("Attribute all_read cannot be modified by user!")





