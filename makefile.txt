main: func_call.c include.h lapack.c main.c makefile my_block.c my.c
	rm -f main
	gcc main.c -o main -I /act/opt/intel/composer_xe_2013.3.163/mkl/include \
	-L /act/opt/intel/composer_xe_2013.3.163/mkl/lib/intel64 \
	-O3 -DMKL_ILP64 -lmkl_avx2 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -m64