# -*- coding: utf-8 -*-

"this code is adapted from: https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab"

"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2024 QUTAC

SPDX-License-Identifier: Apache-2.0

"""
import numpy as np
from scipy.optimize import OptimizeResult

def adam(
    fun,
    x0,
    jac,
    sd,
    #args=(),
    learning_rate=0.01, #upper bound of stepsize in parameter space
    maxiter=1000,
    width=10,
    limit=10,
    sd_limit=10,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    callback=None,
    **kwargs
):
    """
    Parameters
    ----------
        None

    Returns
    ----------
        h : list
            Linear coefficients
        J : dict
            Quadratic coefficients
        final_offset : float
            Offset from the formulation
    """
    
    def get_moving_avg(vals, i, width):
        """
        Parameters
        ----------
            None

        Returns
        ----------
            h : list
                Linear coefficients
            J : dict
                Quadratic coefficients
            final_offset : float
                Offset from the formulation
        """
        avg_i = np.mean(vals[i-width:i])
        std_i = np.std(vals[i-width:i])
        return avg_i, std_i
    
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    success = False
    fvals=[]
    for i in range(startiter, startiter + maxiter):

        g = jac(x)

        if callback and callback(x):
            break
        
        if i==0:
            avg_old=fun(x)

        fvals.append(fun(x))
        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)


        if np.mod(i, int(width))==0:
            avg_new, std = get_moving_avg(fvals, i, width)

            sd_tmp = np.round(sd(x))

            if i > 2 and np.abs(avg_old - avg_new) < limit and np.prod([sd_val > sd_limit for sd_val in sd_tmp]): #all have to be True, so second deriv has to be >0 along all directions
                #print('results:', i, np.abs(avg_old - avg_new), x)
                #print('energy:', fun(x))
                break
            
            avg_old = avg_new

            
        
    
    if i < maxiter-1:
        success = True
        
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=success)



def adam_full(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=0.001,
    maxiter=1000,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from ``autograd/misc/optimizers.py``.

    Parameters
    ----------
        None

    Returns
    ----------
        h : list
            Linear coefficients
        J : dict
            Quadratic coefficients
        final_offset : float
            Offset from the formulation
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        if callback and callback(x):
            break

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)