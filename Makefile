build: homework
homework: ./src/homework.c
	mpicc -o homework ./src/homework.c -lm -Wall
serial: homework
	mpirun -np 1 homework ./in/lenna_bw.pgm out.pgm "blur"
distrib: homework
	mpirun -np 4 homework ./in/lenna_color.pnm 5_out.pmm "blur" "blur"
clean:
	rm -f homework
