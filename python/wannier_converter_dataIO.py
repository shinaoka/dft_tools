"""
I/O class for Wannier converter.
"""
import re
import json as string_converter

from pytriqs.utility import mpi
from pytriqs.applications.dft.messaging import Check, Bracket


class AsciiIO(Check):
    """
        **Class for reading input from ASCII files  **

        **Functionality**::

            (main methods mentioned only please have a look below for all methods)

               -  Reads data from the ASCII file, with case sensitive names
                   (method: read_txt_file)
                
               -  Reads data from the ASCII file honouring Fortran name convention (convention of Wannier90)
                   (method: read_ASCII_fortran_file)



    """

    def __init__(self):
        super(AsciiIO, self).__init__()
        self._verbosity = None


    def read_ASCII_file(self, file_to_read=None, default_dic=None, variable_list_string_val=None):

        """
        Updates  keywords of default_dic by reading data from file_to_read.
        Short characteristics of input format in file_to_read::

                - Only first occurrence of keyword is considered, once keyword is found,
                    entry in the default_dic dictionary with the same name as keyword will be updated.

                - A possible format is accepted: keyword value
                                                 keyword = value.

                - The order of keywords in ASCII file file_to_read is arbitrary.

                - A value can be defined in few lines, provided a criterion
                    of the complete parenthesis is fulfilled.

                - Few pairs keyword value  can be defined in one line.

                - Case of letters matter:
                    Fermi_level
                    is not the same as
                    FERMI_LEVEL.

                - In case of one value which has a form of a single string the following forms are accepted:
                    * key value
                    * key "value"
                    * key 'value'

                    If a value is a list and a string is an element of such a list then double quotations are necessary
                    key =["value1","value2" ...] key1=[1,34, "val", 44] ...

                - If the value is a bool than the following states of such a value are valid:

                    * true
                    * false

                    (please note the lower case of letters in both cases).

                - ! or # mark the beginning of the comment. The comment is ended by the symbol of a new line.

                - Keywords which are obligatory have to have None value as a default value in the default_dic.

                - The first repetition of any keyword is signalized by the error message. Program is terminated.


        :param file_to_read: Text file with valid input parameters to read.
        :type file_to_read: str

        :param default_dic: Dictionary which stores default values of input parameters.
        :type default_dic: dict

        :param variable_list_string_val: list with all keywords which values are expected to be str.
        :type variable_list_string_val:  list of str

        """
        if mpi.is_master_node():
            # basic check of input
            if not isinstance(default_dic, dict):
                self.report_error("Default parameters should have a form of a python dictionary!")

            if isinstance(variable_list_string_val, list):

                if not all([isinstance(variable_list_string_val[i], str)
                            for i in range(len(variable_list_string_val))]):
                    self.report_error("All elements should be of type str!")
            else:
                self.report_error("List was expected!")

            variable_list = list(default_dic.keys())
            variable_list_temp = list(default_dic.keys())
            brackets_inspector = Bracket()  # checks for brackets and stores values
            keyword = ""
            comment_marker = ["!", "#"]
            value_width = -1
            value_over_lines=""
            try:
                with open(file_to_read, "rb") as input_file:
                    lines = input_file.readlines()
                    input_file.close()
                    for count_line, line in enumerate(lines):

                        # remove comments
                        for a in comment_marker:
                            indx = line.find(a)
                            if indx > -1:
                                line = line[:indx]

                        if line.strip():  # ignore blank lines
                            if len(keyword)>0:  value_width += 1
                            words_in_line = line.replace("=", " ").split()

                            # check if valid words over few lines
                            if value_width > 0:
                                if not (len(value_over_lines)>0 and
                                        value_over_lines[0] in brackets_inspector.brackets_patterns.values()):

                                    self.report_error("Every keyword over few lines "
                                                      "must have a value inside brackets!")

                            for value in words_in_line:

                                # check if a new keyword starts
                                if value in variable_list_temp:
                                    keyword = value
                                    value_width = 0 # defines how many lines occupy a value
                                    if not (brackets_inspector.are_brackets_ended() or
                                            brackets_inspector.show_brackets() is None):

                                        self.report_error("Wrong parenthesis  detected in PARAMETERS file. " +
                                                          "Please correct PARAMETERS file and restart the application.")
                                    value_over_lines = ""

                                # case of redefinition
                                elif value in variable_list:

                                    self.report_error("Redefinition of keyword " + value +
                                                            " in PARAMETERS file at line %s. " % (count_line + 1) +
                                                            "Please correct your PARAMETERS file!")
                                    # first line is marked as 1

                                # value to read
                                else:
                                    # loads all brackets to brackets_inspector
                                    brackets_inspector.add_brackets(re.sub(pattern="[a-zA-Z0-9-0]*",
                                                                                    repl=" ",
                                                                                    string=value
                                                                                    ).replace(":", "").
                                                                             replace("'", "").
                                                                             replace(".", "").
                                                                             replace(", ", "").
                                                                             replace("_", "").
                                                                             replace('"', '').
                                                                             replace("+", "").
                                                                             replace("-", "").
                                                                             replace("*", "").
                                                                             replace("/", "").
                                                                             split()
                                                                            )

                                    # explanation:
                                    # re sub
                                    # looks for any letter or number [a-zA-Z0-9-0] and
                                    # substitutes it by empty space: " ". By adding star to *
                                    # the pattern we demand that even if no letter or number is
                                    # found empty space " " will be added between elements of the string
                                    # count sub function parameter is set to its default value 0
                                    # which means that all occurrences which match
                                    # defined pattern will be replaced
                                    #
                                    # replace command remove rest of the content which is not brackets
                                    # then adds brackets in the conserved order to the existing list of
                                    # brackets
                                    #

                                    brackets_inspector.find_parenthesis()
                                    value_over_lines += value

                                # check if the whole value  is read, all value is read when all brackets are paired
                                if ((brackets_inspector.are_brackets_ended() or
                                     brackets_inspector.show_brackets() is None) and not
                                     (keyword == "" or len(value_over_lines) == 0)):

                                    # the special treatment for the string values, no conversion is needed
                                    if keyword in variable_list_string_val:
                                        if len(value_over_lines.split()) == 1:
                                            default_dic[keyword] = value_over_lines

                                        # case of value as a standalone string
                                        else:
                                                default_dic[keyword] = list(value_over_lines.split())

                                    # the case of value which is a list of strings
                                    else:

                                        item = string_converter.loads(value_over_lines)
                                        default_dic[keyword] = item

                                    value_width = 0
                                    variable_list_temp.remove(keyword)  # only first occurrence of the keyword is taken into account
                                    keyword = ""


            except IOError:
                self.report_error("Opening file %s failed!" % file_to_read)

            # Check if all obligatory keywords have been defined
            for key in default_dic:
                if default_dic[key] is None:
                    self.report_error("Not all required input parameters " +
                                      "are defined. %s is not declared!" % key)


            # convert Unicode entries to strings
            for key in default_dic:

                # 1) convert Unicode or strings with additional quotations ("") entries in one value keywords
                if isinstance(default_dic[key], str):
                    str_key = str(default_dic[key]).replace('"', '').replace("'", "")
                    default_dic[key] = str_key
                elif isinstance(default_dic[key], unicode):
                    str_key = str(default_dic[key]).replace("u'", "'").replace("'", "")
                    default_dic[key] = str_key
                # 2) convert Unicode entries  in lists/dictionaries
                elif isinstance(default_dic[key], list) or isinstance(default_dic[key], dict):
                    self.convert_unicode_to_str(default_dic[key])


    def convert_unicode_to_tuple(self, item=None):
        """
        Takes input in the form u"(el_1, el_2,....)"
        and converts it to  (el_1, el_2,....) using json module

        :param item: string to convert into tuple
        :type item: str
        """

        # we actually load string to json which contain list
        assert isinstance(item, unicode) or isinstance(item, str)
        new_decoded_item = string_converter.loads(str(item).replace("'", '"')
                                                           .replace("(", "[")
                                                           .replace(")", "]"))
        decoded_item = []
        for element in new_decoded_item:
            if isinstance(element, unicode):
                decoded_item.append(str(element).replace("u'", "'"))
            else:
                decoded_item.append(element)

        return tuple(decoded_item)  # decode_item is a list, we convert it to a tuple


    def _convert_unicode_to_string_core(self, item=None):
        """
        Convert atom element from unicode to str
        :param item: converts unicode to item
        :return:
        """
        assert isinstance(item, unicode)
        return str(item).replace("u'", "'")


    def convert_unicode_to_str(self, objectToCheck=None):
        """
            Converts unicode to python str, works for nested dicts and lists (recursive algorithm).
            Works for list in which entry is a tuple of items or one item only (item= one string, one float, ...etc. )
            Works for dictionary in which value/keyword is a tuple of items  or one item only
            In case tuple is a  keyword of the dictionary it will be converted to python str with a
            tuple brackets  -> "( )" (keywords in the dictionary are expected to be be
            a string or one of the primitive type of python but not tuple),
            in case of tuple value in dictionary, it will be converted to tuple,
            all unicode will be removed from the entries of tuple.

            :param objectToCheck: Entry in dictionary read by read_txt_file in
                                which  unicode symbols should be converted to str.
            :type  objectToCheck: dict or list or item of them
        """

        if isinstance(objectToCheck, list):
            for i in range(len(objectToCheck)):
                objectToCheck[i] = self.convert_unicode_to_str(objectToCheck[i])

        elif isinstance(objectToCheck, dict):
            for item in objectToCheck:
                if isinstance(item, unicode):

                    decoded_item = self._convert_unicode_to_string_core(item)
                    item_dict = objectToCheck[item]
                    del objectToCheck[item]
                    objectToCheck[decoded_item] = item_dict
                    item = decoded_item

                objectToCheck[item] = self.convert_unicode_to_str(objectToCheck[item])

        # tuple in  a form of string found as value
        elif ((isinstance(objectToCheck, unicode) or isinstance(objectToCheck, str)) and
              '(' in objectToCheck and
              ')' in objectToCheck):

            objectToCheck = self.convert_unicode_to_tuple(objectToCheck)
            self.convert_unicode_to_str(objectToCheck)  # to check if all elements are ok

        # tuple
        elif isinstance(objectToCheck, tuple):
            temp_l = []
            for item in objectToCheck:
                if isinstance(item, unicode):
                    item = self._convert_unicode_to_string_core(item)
                temp_l.append(item)
            objectToCheck = tuple(temp_l)

        # unicode element
        elif isinstance(objectToCheck, unicode):
            objectToCheck = self._convert_unicode_to_string_core(objectToCheck)

        return objectToCheck


    def read_ASCII_fortran_file(self, file_to_read=None, default_dic=None):
        """
        Reads data from ASCII file which is expected to be an input for
        Fortran programs (like Wannier90). Short characteristics of input format in read_txt_fortran_file::
        
            -   Cases of letters in keywords do not matter: Fortran style..
                Cases of value are not checked (all values are read as strings).
                  
            -   All values will be string, later post-processing
                  of each particular keyword-value pair is needed, 
                  to get for example value as python list.).
                  
            -   Can read blocs of data which has the following structure:

                  begin keyword
                  -------------
                  -------------
                  -------------
                  end keyword

                 Cases of letters in special keywords 'begin' and 'end' do not matter. 
                 That is the only way of defining value in few lines.

            -  Apart from bloc structure accepted keyword-value
               definitions are as follows:
                
                keyword value
                
                keyword=value
                 
                keyword = value

                keyword:value

                keyword : value

            -  Exclamation mark  '!' is treated as a beginning of the comment,
               all other symbols are valid part of values or keywords.
                 

        :param file_to_read: Text file with valid input parameters to read.
        :type file_to_read: str

        :param default_dic: Dictionary which stores default values of input parameters.
        :type default_dic: dict


        """

        if mpi.is_master_node():
            # basic check of input
            if not isinstance(default_dic, dict):
                self.report_error("Default parameters should have a form of a python dictionary!")

            variable_list = default_dic.keys()

            # make sure that all keywords are with only lower case letter
            for indx, word in enumerate(variable_list):
                if not variable_list[indx] == word.lower():
                    self.report_error("All keywords should be with only lower case letter!")

            variable_list_temp = list(variable_list)
            in_block = False
            value = ""
            keyword = ""
            comment_marker = ["!", "#"]
            try:
                with open(file_to_read, "r") as input_file:
                    lines = input_file.readlines()
                    input_file.close()
                    for num_line, line in enumerate(lines):
                        # remove comments
                        for a in comment_marker:
                            indx = line.find(a)
                            if indx > -1:
                                line = line[:indx] + "\n"

                        if line.strip():  # ignore blank lines
                            current_line = line.replace("=", " ").replace(":", " ")
                            words_in_line = current_line.split()

                            if words_in_line[0].lower() == "end" and in_block:  # end of block

                                in_block = False
                                if not words_in_line[1] == keyword:
                                    self.report_error("Wrong format of the block!")
                                if not len(words_in_line) == 2:
                                    self.report_error("Wrong format of the block!")


                            elif words_in_line[0].lower() == "begin":  # block for few lines has been found

                                if words_in_line[1].lower() in variable_list:
                                    if words_in_line[1].lower() in variable_list_temp:
                                        keyword = words_in_line[1].lower()
                                        in_block = True
                                        variable_list_temp.remove(keyword)
                                    else:
                                        self.report_warning("Redefinition of keyword in line %s" % (num_line + 1))
                                        # first line is marked as 1

                                if not len(words_in_line) == 2:
                                    self.report_error("Wrong format of the block!")

                            elif in_block:

                                value += line

                            elif words_in_line[0].lower() in variable_list:

                                if words_in_line[0].lower() in variable_list_temp:
                                    keyword = words_in_line[0].lower()
                                    variable_list_temp.remove(keyword)
                                    value = current_line[line.find(words_in_line[0]) + len(keyword):]
                                else:
                                    self.report_warning("Redefinition of keyword in line %s" % (num_line + 1))
                                    # first line is marked as 1
                            if not in_block and keyword != "":
                                default_dic[keyword] = value.strip()
                                value = ""
                                keyword = ""


            except IOError:
                self.report_error("Opening file %s failed!" % file_to_read)


            # Check if all obligatory keywords have been defined
            for key in default_dic:
                if default_dic[key] is None:
                    self.report_error("Not all required input parameters " +
                                      "are defined. %s is not declared!" % key)


            # convert Unicode entries to strings
            for key in default_dic:

                # Convert Unicode or strings with additional quotations ("") entries in one value keywords
                if isinstance(default_dic[key], str):
                    str_key = str(default_dic[key]).replace('"', '').replace("'", "")
                    default_dic[key] = str_key
                elif isinstance(default_dic[key], unicode):
                    str_key = str(default_dic[key]).replace("u'", "'").replace("'", "")
                    default_dic[key] = str_key


    def set_verbosity(self, verbosity=None):
        """

        :param verbosity: new verbosity
        :type verbosity: int

        """
        if not isinstance(verbosity, int):
            if verbosity > 2 or verbosity < 0:
                self.report_error("verbosity parameter has the wrong value. It should be int from [0, 2]")
            self.report_error("verbosity parameter has the wrong value!")
        else:
            self._verbosity = verbosity

    @property
    def verbosity(self):
        """

        :return: current value of verbosity
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, val=None):
        """

        :param val:  potential new value of verbosity

        """
        self.report_error("Verbosity cannot be set this way!")
