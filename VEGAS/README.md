# GPU IMPLEMENTATION OF VEGAS ALGORITHM FOR ADAPTIVE MULTIDIMENSIONAL INTEGRATION by Riccardo Galbiati
# July 20th, 2017
# For information GP Lepage, J. Comput. Phys. 27 (1978) 192.

# VGS3Dimportance.cu is the 3D implementation of VEGAS algorithm with importance sampling.
# The test function used is a 3D Gaussian function whose exact value = 1.
# To compile and launch use nvcc.

# VGS3Dstratified.cu is the 3D implementation of VEGAS algorithm with stratified sampling.
# The test function used is a 3D oscillating function whose exact value = 8.
# To compile and launch use nvcc.

# VEGAS.pdf is a brief review of the main ideas behind the implementation.

# Implementations of different dimensions (from 1 up to 9) are available on request.
# For any information: rgalbiati1994@gmail.com

