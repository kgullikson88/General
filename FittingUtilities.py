"""
Just a set of useful functions that I use often while fitting
So far, just bound functions for leastsq
"""
import numpy



#Define bounding functions:
# lower bound:            lbound(boundary_value, parameter)
# upper bound:            ubound(boundary_value, parameter)
# lower and upper bounds: bound([low, high], parameter)
# fixed parameter:        fixed(fixed_value, parameter)
lbound = lambda p, x: 1e4*numpy.sqrt(p-x) + 1e-3*(p-x) if (x<p) else 0
ubound = lambda p, x: 1e4*numpy.sqrt(x-p) + 1e-3*(x-p) if (x>p) else 0
bound  = lambda p, x: lbound(p[0],x) + ubound(p[1],x)
fixed  = lambda p, x: bound((p,p), x)
