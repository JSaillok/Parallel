hooke_omp: multistart_hooke_omp.c
#	gcc -O3 -fopenmp multistart_hooke_omp.c -o hooke_omp
	gcc -Ofast -march=native -fopenmp multistart_hooke_omp.c -o hooke_omp

hooke_tasks: multistart_hooke_omp_tasks.c
#	gcc -O3 -fopenmp multistart_hooke_omp_tasks.c -o hooke_tasks

	gcc -Ofast -march=native -fopenmp multistart_hooke_omp_tasks.c -o hooke_tasks

hooke_mpi: multistart_hooke_mpi.c
#	mpicc -O3 multistart_hooke_mpi.c -o hooke_mpi

	mpicc -Ofast -march=native multistart_hooke_mpi.c -o hooke_mpi
#	mpirun --use-hwthread-cpus ./hooke_mpi

hooke_hybrid: multistart_hooke_mpi_omp.c

	mpicc -Ofast -fopenmp -march=native multistart_hooke_mpi_omp.c -o hooke_hybrid
# 	run using mpirun --bind-to none -n 2 ./hooke_hybrid
# 	the "--bind-to none" flag is super important !! 

clean:
	rm -f hooke_omp hooke_tasks hooke_mpi hooke_hybrid