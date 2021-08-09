
import numpy as np
import scipy
def complex_euclidean_metric(a1, a2):
    """
        return List value value of euclidean metric for 2 numpy matrixes
        a1 numpy.Array(complex)
        a2 numpy.Array(complex)    
    """
    a1_linear = a1.flatten()
    a2_linear = a2.flatten()
    
    value = np.absolute(np.subtract(a1, a2)).sum()
    return value

def abs_metric(a1, a2):
    """
        return List value value of euclidean metric for 2 numpy matrixes
        a1 numpy.Array(complex)
        a2 numpy.Array(complex)    
    """
    a1_linear = a1.flatten()
    a2_linear = a2.flatten()

    value_r = np.absolute(np.subtract(a1_linear.real, a2_linear.real)).sum()
    value_i = np.absolute(np.subtract(a1_linear.imag, a2_linear.imag)).sum()
    value = value_r + value_i
    return value

def squared_euclidean_metric(a1, a2):
    """
        return List value value of euclidean metric for 2 numpy matrixes
        a1 numpy.Array(complex)
        a2 numpy.Array(complex)    
    """
    a1_linear = a1.flatten()
    a2_linear = a2.flatten()

    value = np.square(np.absolute(np.subtract(a1_linear, a2_linear))).sum()
    return value

"""
    (|Tr(U*txU')|^2/2^m + 1)/(2^m+1)
"""
def trace_metric(a1, a2):
    two_m = a1.shape[0]
    a1 = np.asmatrix(a1)
    a2 = np.asmatrix(a2)
    a1H = a1.getH()
    m_trace = np.trace(np.matmul(a1H, a2))
    value = ((np.absolute(m_trace)**2)/two_m + 1)/(two_m+1)
    return value

def trace_distance(matrix_A, matrix_B):
	matrix_diff = matrix_A - matrix_B
	matrix_diff_dag = numpy.transpose(numpy.conjugate(matrix_diff))
	product = matrix_diff * matrix_diff_dag
	#print "prod= " + str(product)	
	trace_vals = scipy.linalg.eigvals(product)
	return scipy.linalg.norm(trace_vals)

def equality(m1, m2):
    if np.all(np.equal(m1, m2)):
        return 1.0
    else:
        return 0.0

metrics = {
    # "euclidean": euclidean_metric,
    "abs": abs_metric,
    "euclidean": complex_euclidean_metric,
    "squared_euclidean_metric": squared_euclidean_metric,
    "trace_metric": trace_metric,
    "trace_distance": trace_distance,
    "equality": equality
}