# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = k_meansclustering.o 
OBJS = dataset.o filetool.o

# make and compile
kmeans:$(OBJS) $(GPUOBJS)
	$(NVCC) -o kmeans $(OBJS) $(GPUOBJS) 

k_meansclustering.o: k_meansclustering.cu k_meansclustering.h
	$(NVCC) -c k_meansclustering.cu 

dataset.o: dataset.c dataset.h
	$(CXX) -c dataset.c

filetool.o: filetool.c filetool.h
	$(CXX) -c filetool.c

clean:
	rm -f *.o
	rm -f mmul
