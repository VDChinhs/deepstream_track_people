# CUDA_VER?=
# ifeq ($(CUDA_VER),)
# 	$(error "CUDA_VER is not set")
# endif

CUDA_VER := 10.2

CC:= g++

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/glib -I/usr/lib/aarch64-linux-gnu/glib-2.0/include

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= $(wildcard *.cpp)

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo_face.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
