import numpy
from inspect import getframeinfo, currentframe
from mpi4py import MPI

# pytriqs
from pytriqs.utility.mpi import is_master_node, bcast, barrier, slice_array, all_reduce, world,report
from pytriqs.archive import HDFArchive


# ****** Report Class ***************
class Report(object):
    """
        *Simple error handling.*
    """

    def report_error(self, string):
        """
        Error

        :param string: stores the message with the description of the error
        :type string: str
        """
        comm = MPI.COMM_WORLD
        if is_master_node():
            if isinstance(string, str):
                inspect_data = getframeinfo(currentframe().f_back)
                report("Error: " + string +
                   "(file: %s, line: %s, in function: %s)" % (
                inspect_data.filename, inspect_data.lineno, inspect_data.function))
            else:
                myprint_err("Wrong argument of the report_error" +
                        "function. Please send one string as an input parameter!")
        comm.Abort(-1)

    def make_verbose_statement(self,string):
        """
        Statement which can be switched off using verbosity parameter.

        :param string: stores the message with the description of the statement in the form of str
        :type string: str
        """
        if self._verbosity==2:
            if isinstance(string, str):
                report(string)
            else:
                report("Wrong argument of the warning" +
                       "function. Please send one string as an input parameter!")

    def make_statement(self, string):
        """
        Statement

        :param string: stores the message with the description of the statement in the form of str
        :type string: str
        """
        if isinstance(string, str):
            report(string)
        else:
            report("Wrong argument of the warning" +
                   "function. Please send one string as an input parameter!")


    def report_warning(self, string):
        """
        Warning

        :param string: stores the message with the description of the warning in the form of str
        :type string: str
        """
        if self._verbosity==2:
            if isinstance(string, str):
                    report("Warning: " + string)
            else:
                report("Wrong argument of the warning" +
                       "function. Please send one string as an input parameter!")



class Check(Report):
    n_inequiv_corr_shells=None

    def __init__(self):
        """
        *Groups checks which are frequently used.*

        """

        super(Check, self).__init__()
        self._blocnames = [["up", "down"], ["ud"]]
        self._possible_modes = ["r+", "r"]  # r+ is used for "a"  and "w " modes
        self._parameters_changed = False
        self._verbosity=None
        # These parameters have to be set to some valid value by inheriting class
        self.n_corr_shells=None
        self.n_inequiv_corr_shells=None
        self._filename = None
        self._parameters_to_check = None
        self._old_parameters = None
        self.SO = None
        self._n_spin_blocs=None
        self.n_k=None
        self._gf_struct=None
        self._n_spin_blocs=None
        self.n_k=None
        self._beta=None
        self._what_changed=[] #list with parameters which have changed
        self._critical_par_changed=False
        self._critical_par=None

    @property
    def blocnames(self):
        return self._blocnames


    @blocnames.setter
    def blocnames(self,val=None):
        self.report_error("Value of blocnames cannot be  changed!")


    def reset_parameters_changed_attr(self):
        """
        Resets attribute necessary for *check_if_parameters_changed* to function properly.

        """
        self._parameters_changed = False


    def _check_if_parameter_changed(self, parameter_name=None, new_par=None, old_par=None):

        """
        Checks if parameter_name has changed in the dictionary with new parameters.

        :param parameter_name: name of parameter to check
        :type parameter_name: str

        :param new_par: item to check
        :type new_par: dict or list or primitive python type

        :param old_par: item from the previous run  to check
        :type old_par: dict or list or primitive python type
        """

        if isinstance(new_par, list):
            if isinstance(old_par, list):
                if len(new_par) == len(old_par):
                    for item in range(len(new_par)):
                        if isinstance(new_par[item],numpy.ndarray):
                            self._check_if_parameter_changed(parameter_name=parameter_name,
                                                         new_par=new_par[item],
                                                         old_par=old_par[item])
                        elif new_par[item] in old_par:
                            self._check_if_parameter_changed(parameter_name=parameter_name,
                                                         new_par=new_par[item],
                                                         old_par=old_par[item])
                        else:
                            self.report_par_change(item=parameter_name)
                            break

                else:

                    self.report_par_change(item=parameter_name)

            else:

                self.report_par_change(item=parameter_name)

        elif isinstance(new_par, dict):
            if isinstance(old_par, dict):
                if len(new_par) == len(old_par):
                    for item in new_par:
                        if isinstance(item,numpy.ndarray):
                            self._check_if_parameter_changed(parameter_name=parameter_name,
                                                             new_par=new_par[item],
                                                             old_par=old_par[item])

                        elif item in old_par:
                            self._check_if_parameter_changed(parameter_name=parameter_name,
                                                             new_par=new_par[item],
                                                             old_par=old_par[item])
                        else:
                            self.report_par_change(item=parameter_name)
                            break
                else:
                    self.report_par_change(item=parameter_name)

            else:

                self.report_par_change(item=parameter_name)

        elif isinstance(new_par, numpy.ndarray):
            if isinstance(old_par, numpy.ndarray):
                if not ( new_par.shape==old_par.shape and numpy.allclose(new_par, old_par)):

                    self.report_par_change(item=parameter_name)

            else:
                self.report_par_change(item=parameter_name)

        elif (new_par != old_par and not
        (new_par is None or old_par == "None")):

            self.report_par_change(item=parameter_name)


    @property
    def parameters_changed(self):

        return self._parameters_changed


    @parameters_changed.setter
    def parameters_changed(self,val):
        self.report_error("Attribute parameters_changed cannot be changed by user!")


    def check_parameters_changed(self, dictionary=None,hdf_dir=None):
        """
        Checks if parameters in sumk_dft have changed


        :param dictionary:  dictionary which stores key-value pairs which
                            will be compared with previous data  from hdf file
        :type dictionary: dict

        :param hdf_dir: name of the directory in hdf file where is the  dictionary to compare
        :type hdf_dir: str
        """

        if dictionary is None or not isinstance(dictionary,dict):
            self.report_error("Define dictionary with data to compare")

        # function check_if_parameters_changed from Check class which was inherited by Wannier2TRIQS
        self.reset_parameters_changed_attr()
        self._parameters_changed_core( items_to_check=dictionary.keys(),
                                          dictionary=dictionary,
                                          hdf_dir=hdf_dir)

        if self._parameters_changed:

            if self._critical_par_changed:
                for item in self._what_changed:
                    if item in self._critical_par:
                        self.make_statement("Critical parameter %s has changed since the "
                                            "last run. Please correct your input file!"%item)
                self.report_error("Invalid input data program aborted. Please correct input and rerun.")

            else:
                self._update_par_hdf(name=hdf_dir,
                                 dictionary=dictionary)


    def _save_par_hdf(self,name=None,dictionary=None):
        """
        Saves data to hdf file.

        :param name: Name of the folder in hdf file where data will be saved, expects name from the main "directory",
                    if name is not present it will be created
        :type name: str

        :param dictionary:  dictionary with crucial data to save, if
                            any entry is already present in hdf file it will be overwritten, mandatory parameter
        :type dictionary: dict

        """
        if is_master_node:

             try:

                ar = HDFArchive(self._filename + ".h5", "a")


                if not (name in ar):

                    self.report_warning("""Directory %s not found."""%name)
                    ar.create_group(name)



                for it in dictionary:
                    if it in ar[name]and not self._parameters_changed:
                        self.report_warning("Element "+it+" found in %s folder. Its previous content will be overwritten."%name)
                    ar[name][it] = dictionary[it]

                del ar

             except IOError:

                self.report_error("Appending "+ self._filename + ".h5 file failed!")


    def _parameters_changed_core(self, items_to_check=None,dictionary=None, hdf_dir=None):
        """

        Checks if parameters have changed -- core function.
        Private method of check class should be used
        together with check_parameters_changed
        Compares dictionary with old data and
        dictionary with new data

        :param items_to_check: list of items to check
        :type items_to_check: list of str

        :param dictionary: dictionary with values to check
        :type dictionary: dict

        :param hdf_dir: name of the directory in hdf file where is the  dictionary to compare,
                        it is expected to be in the main directory of hdf file
        :type hdf_dir: str

        :return: True if parameters have changed, False otherwise
        """
        self._parameters_to_check=dictionary
        if is_master_node():
            try:
                ar = HDFArchive(self._filename + ".h5", "a")
                if hdf_dir in ar:
                    old_parameters={}
                    for item in items_to_check:
                        if item in ar[hdf_dir]:
                            old_parameters[item] = ar[hdf_dir][item]
                        else:
                            self._parameters_changed = True
                            self.report_warning("Keyword %s not found in hdf file. New input "
                                                "parameter will be used in calculation (update of software?)"%item)
                            del ar
                            return self._parameters_changed
                    self._old_parameters = old_parameters
                    for item in items_to_check:
                        if item in self._parameters_to_check:
                            if item in self._old_parameters:

                                self._check_if_parameter_changed(parameter_name=item,
                                                                 new_par=self._parameters_to_check[item],
                                                                 old_par=self._old_parameters[item])
                            else:
                                self.report_warning("Item %s not found in old parameters!" % item)
                        else:
                            self.report_warning("Item %s not found in the current parameters" % item)

                else:

                    self._parameters_changed = True

                del ar
            except IOError:

                self.report_error("Data from  file " + self._filename + ".h5 couldn't be read")

        self._parameters_changed=bcast(self._parameters_changed)

        return self._parameters_changed


    def _convert_str_to_None(self, par=None):
        """
        Converts "None" to None.

        :param par: dictionary or item of dictionary
        :type par: dict
        """
        if isinstance(par, list):
            for item in range(len(par)):
                if par[item] == "None":
                    par[item] = None
                elif (    isinstance(par[item], list) or
                              isinstance(par[item], dict)):
                    self._convert_str_to_None(par[item])

        elif isinstance(par, dict):
            for item in par:
                if par[item] == "None":
                    par[item] = None
                elif (    isinstance(par[item], list) or
                              isinstance(par[item], dict)):
                    self._convert_str_to_None(par[item])


    def _convert_None_to_str(self, par=None):
        """
        Converts None values to "None".

        :param par: dictionary or item of dictionary
        :type par: dict
        """
        if isinstance(par, list):
            for item in range(len(par)):
                if par[item] is None:
                    par[item] = "None"
                elif (    isinstance(par[item], list) or
                              isinstance(par[item], dict)):
                    self._convert_None_to_str(par[item])

        elif isinstance(par, dict):
            for item in par:
                if par[item] is None:
                    par[item] = "None"
                elif (    isinstance(par[item], list) or
                              isinstance(par[item], dict)):
                    self._convert_None_to_str(par[item])


    def _update_par_hdf(self,name=None,dictionary=None):
        """
            Updates data in  hdf file.

            :param name: Name of the folder in hdf file where data will be updated, expects name from the main "directory"
            :type name: str

            :param dictionary:  dictionary with crucial data to update
            :type dictionary: dict

        """
        if is_master_node():
            try :
                ar = HDFArchive(self._filename + ".h5", "a")
                if not name in ar: ar.create_group(name)
                ar[name].update(dictionary)

                del ar
            except IOError:
                self.report_error("Appending to file "+self._filename + ".h5 failed!")


    def check_hdf_file(self, hdf_file=None, mode=None):

        """
        Checks if the hdf file is valid and if its mode is correct.

        :param hdf_file: hdf file to check
        :type hdf_file: HDFArchive

        :param mode: mode in which hdf file should be accessed
        :type mode: str

        :return: True if hdf_file is valid, False otherwise
        """
        result = True
        if not mode in self._possible_modes:
            self.report_error("Wrong mode. Mode is %s but should be %s " % (hdf_file._group.mode, mode))

        if not isinstance(hdf_file, HDFArchive):
            result = False
        else:
            if not hdf_file._group.mode == mode:
                result = False

        return result


    def check_shell(self, x=None, t=None):
        """
        Checks if shell has a correct structure,

        :param x: shell to  check
        :type x: dictionary

        :param t: list of keywords which should be in the shell
        :type t: str

        :return: True if the structure of shell  is correct otherwise False
                 Structure of shell is considered to be correct if keywords of
                 x are equal to t and  for each key-value in x all values are of type int

        """

        return isinstance(x, dict) and  \
               isinstance(t, list) and \
               all([isinstance(key,str) for key in t ]) and \
               sorted(t)==sorted(x.keys()) \
               and all([isinstance(x[key],int) for key in x])


    def check_n_k(self,n_k=None):
        return 0<n_k<self.n_k


    def check_n_spin_bloc(self,n_spin_bloc=None):
        return 0<=n_spin_bloc<self._n_spin_blocs


    def check_n_corr(self,n_corr=None):
        """
        Checks whether number of correlated shell is valid.
        :param num_corr: number of correlated shell
        :type num_corr: int

        :return: bool, True if num_corr is valid otherwise False
        """
        return (isinstance(n_corr, int) and
              0 <= n_corr < self.n_corr_shells)


    def check_inequivalent_corr(self, n_corr=None):

        """
        Checks whether number of an inequivalent correlated shell is valid.

        :param n_corr: number of inequivalent correlated shell to be checked
        :type n_corr: int

        :return: bool, True if num_corr is valid otherwise it is False.
        """
        if hasattr(self,"_n_inequiv_corr_shells"):

            return  (isinstance(n_corr, int) and
                     0 <= n_corr < self.n_inequiv_corr_shells)
        else:

            return  (isinstance(n_corr, int) and
                     0 <= n_corr < self.__class__.n_inequiv_corr_shells)


    def report_par_change(self, item=None):
        """
        Makes a report  about the parameter which has changed.

        :param new_parameter: new dictionary with parameters to check
        :type: new_parameter: HDFArchive


        """

        if isinstance(item,str):
            self.report_warning("Previously parameter "+item+
                            " was set to %s. Now it is %s."
                            % (self._old_parameters[item], self._parameters_to_check[item]))
        else:
            self.report_warning("Previously parameter %s"
                            " was set to %s. Now it is %s."
                            % (item, self._old_parameters[item], self._parameters_to_check[item]))


        self._parameters_changed = True
        if item in self._critical_par: self._critical_par_changed=True
        self._what_changed.append(item)


    def check_Gf_obj(self, Object=None, n_corr=None):
        """Checks if the structure of Green's function object is correct.

        :param Object:  Green's function object for the particular inequivalent correlated shell which will be checked
        :type  Object: BlockGf which consist of GfImFreq block(s)

        :param n_corr: number of inequivalent correlated shell
        :type n_corr: int

        :return: True if Gf object is valid, otherwise will stop with error
        """

        if not self.check_inequivalent_corr(n_corr=n_corr):
             self.report_error("Wrong value of inequivalent correlated shell!")

        if not isinstance(Object, BlockGf):
             self.report_error("Wrong Object argument. Green's function object was expected!")

        for  struct in Object:
            bloc=struct[0]

            if not bloc in self._gf_struct:
                self.report_error("Wrong structure of Green's function (wrong blocks)")

            if not isinstance(Object[bloc], GfImFreq):
                self.report_error("Wrong structure of Green's function (wrong type)")

            str_indices=["%s"%i for i in range(len(self._gf_struct[bloc]))]
            if str_indices!=Object[bloc].indices:
                self.report_error("Wrong structure of Green's function (wrong indices)")

            if Object[bloc].beta != self._beta:
                self.report_error("Invalid value of beta for block %s"%bloc)

        return True

    def check_T_n_corr(self,T_array=None):
        """
        Checks matrix T.

        :param T_array: T matrix to check for one particular inequivalent correlated shell
        :type T_array: numpy.ndarray

        :return T_array if T_array is valid otherwise None
        :rtype : numpy.ndarray or None

        """

        if T_array is None:
            return None

        if isinstance(T_array, numpy.ndarray):
            block_length = len(T_array)
            #check T structure
            for i in range(block_length):
                if len(T_array[i]) == block_length:
                    if not all([isinstance(T_array[i,j],
                           complex) for j in range(block_length)]):

                        self.report_warning("Each entry must be of complex type!"
                                            " Default matrix T will be build for %s "
                                            "inequivalent correlated shell." % self._n_corr)
                        return None
                else:
                    self.report_warning("Square matrix was expected! "
                                        "Default matrix T will be build for %s "
                                        "inequivalent correlated shell." % self._n_corr)
                    return None

            if self._verbosity == 2:

                self.make_statement("Valid T matrix found")
                self.show_obj(obj=T_array,
                              name="T matrix for %s inequivalent correlated shell" % self._n_corr)

            return T_array

        else:

            self.report_warning("Wrong structure of T! Default matrix T "
                                "will be build for %s inequivalent "
                                "correlated shell." % self._n_corr)
            return None


    def show_obj(self, obj=None, name=None):
        """
        Shows any object (provide its string representation is implemented).

        :param obj: python object
        :type obj: Everything which can be send as an argument to print function.

        :param name: name of object
        :type name: str
        """
        if not isinstance(name, str):
            report("Wrong name argument for the function" +
                   " . Please send one string as a name parameter!")

        if is_master_node:
            print "%s has the following form: \n" % name
            print obj

