import numpy as np

def check_class(cl):
    """
    Wrapper that ensurese the setter method is given the proper type.
    """
    def wrapper(func):
        def checked(self, cl_to_check):
            if not isinstance(cl_to_check, cl):
                msg = "%s is not type %s" %\
                (cl_to_check.__class__.__name__, cl.__name__)
                raise RuntimeError(msg)
            func(self, cl_to_check)
        return checked
    return wrapper

def class_dict(cl):
    '''
    Cycle throught each attribute of a class and store information
    in a dictionary. The key will be the class name and value if
    that class has been set.
    '''
    dic = {}
    for attr_name, attr in cl.__dict__.iteritems():
        comp = getattr(cl, attr_name)
        if isinstance(comp, (int, float, str,
            dict, tuple, list, np.ndarray)):
            continue
        if comp is None:
            dic[attr_name] =  'Not Defined'
        else:
            dic[attr_name] = attr.__class__.__name__
    return dic

#def _check_component(self):
#    '''
#    Cycle through all classes and make sure all attributes are
#    set. If any attributes are not set then raise and exception.
#    '''
#
#    for attr_name, cl in self.__dict__.iteritems():
#        comp = getattr(self, attr_name)
#        if isinstance(comp, (int, float, str)):
#            continue
#        if comp == None:
#            raise RuntimeError("Component: %s not set." % attr_name)

#def _create_components_timeshot(self):
#    '''
#    Cycle through all classes and record all attributes and store
#    it in a dictionary, key = class name val = attributes.
#    '''
#    dict_output = {}
#    for attr_name, cl in self.__dict__.iteritems():
#
#        comp = getattr(self, attr_name)
#
#        # ignore parameters
#        if isinstance(comp, (int, float, str)):
#            continue
#
#        # store components
#        d = {}
#        for i in dir(cl):
#            x = getattr(cl, i)
#            if isinstance(x, (int, float, str)):
#                d[i] = x
#
#        dict_output[attr_name] = (cl.__class__.__name__, d)
#
#    return dict_output
#
#def _create_components_from_dict(self, cl_dict):
#    '''
#    Create a simulation from dictionary cl_dict created by _create_components_from_dict
#    '''
#    for comp_name, (cl_name, cl_param) in cl_dict.iteritems():
#        cl = getattr(phd, cl_name)
#        x = cl(**cl_param)
#        setattr(self, comp_name, x)
