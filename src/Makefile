PROGRAM = remesh

CU_FILES = cudaRemesh.cu

CPP_FILES = main.cpp mesh.cpp test.cpp seqRemesh.cpp


LDFLAGS = -L/usr/include/c++/11/ -L/usr/local/cuda-11.7/lib64/ -lcudart

NVCC_FLAGS = -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/g++-11
NVCC = /usr/local/cuda-11.7/bin/nvcc

CXX_FLAGS = -O3 -Wall -g
CXX = /usr/bin/g++-11 -m64

OBJ_DIR = objs

CU_OBJS = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(CU_FILES))
CPP_OBJS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(CPP_FILES))
# CPP_OBJS = $(CPP_FILES:.cpp=.o)

all: $(PROGRAM)

clean:
	rm -rf $(PROGRAM) $(OBJ_DIR)/*.o

$(PROGRAM): $(CU_OBJS) $(CPP_OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) -L/usr/include/c++/11/ -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)