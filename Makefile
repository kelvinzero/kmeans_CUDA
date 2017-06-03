# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

OPT = -arch=sm_35 -rdc=true -lcudadevrt
# here are all the objects
GPUOBJS = k_meansclustering.o clusterdata.cu kmeanskernel.o
 
OBJS = dataset.o filetool.o

# make and compile
kmeans:$(OBJS) $(GPUOBJS)
	$(NVCC) $(OPT) -o kmeans $(OBJS) $(GPUOBJS) 

k_meansclustering.o: k_meansclustering.cu k_meansclustering.h clusterdata.o
	$(NVCC) $(OPT) -c k_meansclustering.cu 

kmeanskernel.o : kmeanskernel.cu kmeanskernel.h
	$(NVCC) $(OPT) -c kmeanskernel.cu 	

clusterdata.o : clusterdata.cu clusterdata.h
	$(NVCC) -c clusterdata.cu 	

dataset.o: dataset.c dataset.h
	$(CXX) -c dataset.c

filetool.o: filetool.c filetool.h
	$(CXX) -c filetool.c

clean:
	rm -f *.o
	rm -f mmul
