# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
OPT = -arch=sm_35 -rdc=true -lcudadevrt
COPT = -lm
# here are all the objects
GPUOBJS = clusterkernel.o k_meansclustering.o clusterdata.cu #kmeanskernel.o
 
OBJS = dataset.o filetool.o kmeanscpu.o timing.o

# make and compile
kmeans: $(OBJS) $(GPUOBJS) 
	$(NVCC) $(OPT) -o kmeans $(OBJS) $(GPUOBJS) 
	
run: clean kmeans
	./kmeans genedata.ssv clusterout 10 1

k_meansclustering.o: k_meansclustering.cu k_meansclustering.h clusterdata.o
	$(NVCC) $(OPT) -c k_meansclustering.cu 

clusterkernel.o: clusterkernel.cu clusterkernel.h
	$(NVCC) $(OPT) -c clusterkernel.cu

timing.o : timing.c timing.h
	$(CXX) $(COPT) -c timing.c

kmeanscpu.o : kmeanscpu.c kmeanscpu.h
	$(CXX) $(COPT) -c kmeanscpu.c

kmeanskernel.o : kmeanskernel.cu kmeanskernel.h
	$(NVCC) $(OPT) -c kmeanskernel.cu 	

clusterdata.o : clusterdata.cu clusterdata.h
	$(NVCC) -c clusterdata.cu 	

dataset.o: dataset.c dataset.h
	$(CXX) -c dataset.c

filetool.o: filetool.c filetool.h
	$(CXX) -c filetool.c

shrinkfile: shrinkfile.c
	$(CXX) -o shrinkfile shrinkfile.c

clean:
	clear
	rm -f *.o
	rm -f kmeans
	rm -f shrinkfile
	rm -f *.CL* 
