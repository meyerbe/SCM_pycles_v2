#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from ThermodynamicsDry cimport ThermodynamicsDry
from ThermodynamicsSA cimport ThermodynamicsSA
import numpy as np
cimport numpy as np
# cimport Grid
# cimport PrognosticVariables


from scipy.integrate import odeint

include 'parameters.pxi'

# def ThermodynamicsFactory(namelist, LatentHeat LH):
def ThermodynamicsFactory(namelist):
    type = namelist['microphysics']['scheme']
    if(type == 'None_Dry'):
        print('Thermodynamics: dry')
        # pass
        # return ThermodynamicsDry(namelist,LH)
        return ThermodynamicsDry(namelist)
    elif(type == 'None_SA' or type == 'None_Dry'):
        print('Thermodynamics: moist')
        # pass
        # return ThermodynamicsSA(namelist,LH)
        return ThermodynamicsSA(namelist)
