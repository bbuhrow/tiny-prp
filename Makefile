# MIT License
# 
# Copyright (c) 2025, Ben Buhrow
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.
# 
# 

CC = clang
CFLAGS = -fno-common -g -m64 -std=gnu99
WARN_FLAGS = -Wall -Wconversion
OPT_FLAGS = -O2 -DNDEBUG 
OBJ_EXT = .o

# ===================== path options =============================

# standard search directories for headers/libraries within yafu.
# These should normally not be modified.
INC = -I. 
LIBS = -L. 

# we require additional search directories for gmp
# for libraries and headers.  Change
# these if your installation locations differ.
INC += -I/users/buhrow/src/c/gmp_install/6.2.0-aocc/include
LIBS += -L/users/buhrow/src/c/gmp_install/6.2.0-aocc/lib


# ===================== compiler options =========================

ifeq ($(COMPILER),icc)
	CC = icc
	INC += -L/usr/lib/gcc/x86_64-redhat-linux/4.4.4
	CFLAGS += -qopt-report=5
endif


# ===================== architecture features =========================
# several functions in can take advantage of advanced processor
# features (instruction sets).  Specify on the command line, e.g., 
# USE_AVX2=1

ifeq ($(ICELAKE),1)
	CFLAGS += -DUSE_BMI2 -DUSE_AVX2 -DUSE_AVX512F -DUSE_AVX512BW -DSKYLAKEX -DIFMA -march=icelake-client
	SKYLAKEX = 1
else

ifeq ($(SKYLAKEX),1)
	CFLAGS += -DUSE_BMI2 -DUSE_AVX2 -DUSE_AVX512F -DUSE_AVX512BW -DSKYLAKEX -march=skylake-avx512 
endif
	
endif

ifeq ($(COMPILER),icc)
	LIBS +=  -lsvml
endif

CFLAGS += $(OPT_FLAGS) $(WARN_FLAGS) $(INC)



#--------------------------- file lists -------------------------
PRP_SRCS = \
	tinyprp.c \
	monty.c


PRP_OBJS = $(PRP_SRCS:.c=$(OBJ_EXT))


#---------------------------Header file lists -------------------------
COMMON_HEAD = monty.h \
	tinyprp.h


#---------------------------Make Targets -------------------------

help:
	@echo "to build tinyprp:"
	@echo "make tinyprp"
	@echo "add 'SKYLAKEX=1' to include AVX512F,VL,DQ,BW vectorization instruction support"
	@echo "add 'ICELAKE=1' to include SKYLAKEX + AVX512IFMA vectorization instruction support"
	@echo ""
	@echo "add COMPILER=icc to build with the icc compiler (if installed)"



tinyprp: $(PRP_OBJS)
	$(CC) $(CFLAGS) $(PRP_OBJS) -o tinyprp $(LIBS) -lgmp -pthread

clean:
	rm -f $(PRP_OBJS)

#---------------------------Build Rules -------------------------

%.o: %.c $(COMMON_HEAD)
	$(CC) $(CFLAGS) -c -o $@ $<
	
