
EXECUTABLE := chessCuda

CU_FILES   := mini.cu

CU_DEPS    :=

CC_FILES   := *.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64 -std=c++11
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35


OBJS=$(OBJDIR)/playchess.o  $(OBJDIR)/mini.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
