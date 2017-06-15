K-Means GPU clustering on NVIDIA GPU using CUDA

./kmeans <inputfile> <outputfile prefix> <k clusters> <1 GPU / 0 CPU>

Input file format is expected to be space separated values.
Non-numeric attributes are loaded as zero values.
K cluster files are written using the outfile prefix plus .CL#.
Using 1 for GPU will perform all cluster operations on the GPU.
Using 0 for CPU will perform all cluster operations on the CPU (no parallelization).

To test with included files:

./kmeans genedata.ssv clusterout 12 1 

	-Values for clusterout and k = 12 can be adjusted.




To use the gnuplot converter:

./gnuconvert <inputfile> <outputfile>

Gnuconvert will use any cluster output file and parse it to a usable gnuplot coordinates file.

