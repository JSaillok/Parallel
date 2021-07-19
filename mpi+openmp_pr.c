#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

#define MAXVARS		(250)	/* max # of variables	     */
#define RHO_BEGIN	(0.5)	/* stepsize geometric shrink */
#define EPSMIN		(1E-6)	/* ending value of stepsize  */
#define IMAX		(5000)	/* max # of iterations	     */

/* global variables 
 *
 * In the hybrid version (OpenMP threads inside MPI nodes), we once again need to protect 
 * the funevals, global variable from concurrent writes from the OMP threads.
 * Thus, the same practice is followed, we are going to give each thread is own **threadprivate**
 * copy of the variable, and NOT just a private copy, since the don't persist accross regions 
 * as they are stack allocated, and not heap allocated like the threadprivate variables, are.
 */
unsigned long funevals = 0;
#pragma omp threadprivate(funevals)

/* Each MPI node's thread will add its value of funevals, into this variable using an atomic operation.
 * Therefore, this variable will store the total number of function evaluations for each MPI node,
 * which will, of course, be equal to the sum of the funevals values of its OMP threads.
 */
unsigned long node_funevals = 0; 

/* Rosenbrocks classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    	double fv;
    	int i;

	funevals++;
    	fv = 0.0;
    	for (i=0; i<n-1; i++)   /* rosenbrock */
        	fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);

    	return fv;
}

/* given a point, look for a better one nearby, one coord at a time */
double best_nearby(double delta[MAXVARS], double point[MAXVARS], double prevbest, int nvars)
{
	double z[MAXVARS];
	double minf, ftmp;
	int i;
	minf = prevbest;
	for (i = 0; i < nvars; i++)
		z[i] = point[i];
	for (i = 0; i < nvars; i++)
	{
		z[i] = point[i] + delta[i];
		ftmp = f(z, nvars);
		if (ftmp < minf)
			minf = ftmp;
		else
		{
			delta[i] = 0.0 - delta[i];
			z[i] = point[i] + delta[i];
			ftmp = f(z, nvars);
			if (ftmp < minf)
				minf = ftmp;
			else
				z[i] = point[i];
		}
	}
	for (i = 0; i < nvars; i++)
		point[i] = z[i];

	return (minf);
}


int hooke(int nvars, double startpt[MAXVARS], double endpt[MAXVARS], double rho, double epsilon, int itermax)
{
	double delta[MAXVARS];
	double newf, fbefore, steplength, tmp;
	double xbefore[MAXVARS], newx[MAXVARS];
	int i, j, keep;
	int iters, iadj;

	for (i = 0; i < nvars; i++)
	{
		newx[i] = xbefore[i] = startpt[i];
		delta[i] = fabs(startpt[i] * rho);
		if (delta[i] == 0.0)
			delta[i] = rho;
	}
	iadj = 0;
	steplength = rho;
	iters = 0;
	fbefore = f(newx, nvars);
	newf = fbefore;
	while ((iters < itermax) && (steplength > epsilon))
	{
		iters++;
		iadj++;
#if DEBUG 
		printf("\nAfter %5d funevals, f(x) =  %.4le at\n", funevals, fbefore);
		for (j = 0; j < nvars; j++)
			printf("   x[%2d] = %.4le\n", j, xbefore[j]);
#endif
		/* find best new point, one coord at a time */
		for (i = 0; i < nvars; i++) {
			newx[i] = xbefore[i];
		}
		newf = best_nearby(delta, newx, fbefore, nvars);
		/* if we made some improvements, pursue that direction */
		keep = 1;
		while ((newf < fbefore) && (keep == 1))
		{
			iadj = 0;
			for (i = 0; i < nvars; i++)
			{
				/* firstly, arrange the sign of delta[] */
				if (newx[i] <= xbefore[i])
					delta[i] = 0.0 - fabs(delta[i]);
				else
					delta[i] = fabs(delta[i]);
				/* now, move further in this direction */
				tmp = xbefore[i];
				xbefore[i] = newx[i];
				newx[i] = newx[i] + newx[i] - tmp;
			}
			fbefore = newf;
			newf = best_nearby(delta, newx, fbefore, nvars);
			/* if the further (optimistic) move was bad.... */
			if (newf >= fbefore)
				break;

			/* make sure that the differences between the new */
			/* and the old points are due to actual */
			/* displacements; beware of roundoff errors that */
			/* might cause newf < fbefore */
			keep = 0;
			for (i = 0; i < nvars; i++)
			{
				keep = 1;
				if (fabs(newx[i] - xbefore[i]) > (0.5 * fabs(delta[i])))
					break;
				else
					keep = 0;
			}
		}
		if ((steplength >= epsilon) && (newf >= fbefore))
		{
			steplength = steplength * rho;
			for (i = 0; i < nvars; i++)
			{
				delta[i] *= rho;
			}
		}
	}
	for (i = 0; i < nvars; i++)
		endpt[i] = xbefore[i];

	return (iters);
}

/* Each process will make its OMP threads, test several randomly generated
 * starting points to find the one that minimizes the Rosenbrock objective function.
 * 
 * After a per-process reduction to find each node's optimal variables,
 * each rank sends its findings to the master rank (rank 0), which
 * checks whether the value it received from rank i (where i in [1,nprocs-1]), is 
 * better than the current total minimum value, and if so, it updates the corresponding variables.
 * 
 * Finally, there is a reduction on the funevals variable from each node, to get the total number
 * of times that the Rosenbrock function was called
 */
int main(int argc, char *argv[])
{       
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if(provided < MPI_THREAD_MULTIPLE)      // check if MPI threading support is the one required
        {
                fprintf(stderr, "Error : MPI inadequate threading support\n");
                MPI_Finalize();
                return 1;
        }
        int rank, nprocs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

        int hwthreads = omp_get_num_procs();

        /* Rank 0 will check whether the number of desired MPI nodes/processes is greater 
         * than the number of available hardware threads, i.e the number of virtual cores.
         * If so, processor oversubscription is going to take place, thus significantly 
         * degrading the program's performance.
         *
         * We also need to check whether the number of MPI processes desired is a multiple
         * of the number of available hardware threads. This is needed in order to allow
         * each MPI process to run with the same number of hardware threads, in order to
         * achieve (near) uniform load balancing amongst all MPI nodes.
         * 
         * If any of the above is true, then rank 0, calls MPI_Abort which terminates all
         * MPI processes, and returns the specified exit code, which in this case, is set to 1.
         * 
         * By enabling the user to specify the desired number of MPI nodes/processes, we are
         * allowing the program to be more flexible with its number of MPI nodes.
         */
        if(rank == 0)
        {
                if(nprocs > hwthreads)
                {
                        fprintf(stderr, "\n************************************* \
                                         \nERROR : number of MPI nodes greater than the number of available hardware threads\n\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                }                        

                if(hwthreads % nprocs)
                {
                        fprintf(stderr, "\n************************************* \
                                         \nERROR: Number of MPI Nodes must be a multiple of the number of available hardware threads\n\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                }
        }

        /* All processes need to wait for rank 0 to finish checking whether the program should
         * be allowed to run or not, before continuing 
         */
        MPI_Barrier(MPI_COMM_WORLD);

        // disable OMP dynamic mode, as we strictly want the specified number of threads to run, and NOT less
        omp_set_dynamic(0);     
     
        /* Using the following omp_set_num_threads() call, we allow each MPI process to have
         * the same number of hardware threads.
         *
         * The program will run correctly, if the number of MPI nodes is less than the number of
         * hardware threads (see above). 
         *
         * However, it is recommended to run the program using **ONLY 2** MPI processes (run using mpirun -n 2),
         * in order to minimize communication between processes, and thus the resulting overhead.
         *
         * Each process will have **half the number of hardware threads** OMP threads.
         * Since the number of hardware threads on all modern CPUs, is a power of 2, this division yields no errors.
         * 
         * For example, in a CPU with 12 hardware threads, we are going to run the programm as 
         * 2 MPI Processes, with 6 OMP threads each.
         * 
         * If the programm is run without specifying the desired number of MPI nodes
         * (i.e without using the -n flag), the default behavior is to run the program
         * with (#hardware threads / 2) MPI Processes.
         */
        omp_set_num_threads(hwthreads / nprocs);

#if DEBUG
        int t = hwthreads / nprocs;
        printf("I am rank %d, running with %d threads\n", rank, t);
#endif

	int itermax = IMAX;
	double rho = RHO_BEGIN;
	double epsilon = EPSMIN;
	int nvars;
	int trial, ntrials;
	double fx;
	int i, jj;
	double t0, t1;

	double best_fx = 1e10;
	double best_pt[MAXVARS];
	int best_trial = -1;
	int best_jj = -1;

	for (i = 0; i < MAXVARS; i++) best_pt[i] = 0.0;

	ntrials = 128  * 1024;	/* number of trials */
	nvars = 16;		/* number of variables (problem dimension) */
	srand48(time(0));

        long tseed = time(NULL);

        /* Each rank/MPI process will be responsible for executing a part of the total trials.
         * Since we want to run 2 MPI Processes, each node will be responsible for executing
         * a half of the total trials.
         */
        int rank_ntrials = ntrials / nprocs;

        /* We need a barrier here, in order to be sure that all processes are done setting up their 
         * variables (i.e. their data environment), and are ready to proceed into their computational for-loop.
         * This is needed, in order to get the real starting time of the parallel computations.
         * 
         * If we didn't use a barrier here, we could end up getting a wrong starting time
         */
        MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0)
                t0 = MPI_Wtime();

        /* Initiate a parallel region within a MPI Process (i.e spawn each processe's OMP threads).
         * Each thread will act the same way as the OMP worksharing version of the program.
         * Each MPI node, has to execute **rank_ntrials** total trials. These trials are
         * distributed between each OMP threads, using the #pragma omp for directive.
         * As in the other OMP versions, we use the dynamic loop scheduling policy, which
         * we know is better when all loop iterations have different completion times, and
         * which was also observed to provide a non-negligible speedup compared to the
         * default loop scheduling policy, i.e the static one.
         * 
         * Thus, each thread constructs a random point in the 16-dimensional space and calls hooke.
         * It then checks, whether its value of the objective function, is less than its node's
         * minimum value, and if so updates the appropriate variables in an atomic manner and the 
         * best_pt vector using a critical region. 
         * Finally, each thread atomically adds its thread-local value of the funevals variable, into the
         * node_funevals variable, which stores the total number of calls to the Rosenbrock function, made
         * by each node (i.e from the node's threads).
         */
        #pragma omp parallel 
        {
                unsigned short buffer[3];
                buffer[0] = 0;
                buffer[1] = 0;
                buffer[2] = tseed + omp_get_thread_num();

                double local_fx = 0;
                double startpt[MAXVARS], endpt[MAXVARS];
                int local_jj = 0;

                #pragma omp for schedule(dynamic) nowait
                for(trial  = 0; trial < rank_ntrials; trial++)
                {
                        /* starting guess for rosenbrock test function, search space in [-4, 4) */
                        for (i = 0; i < nvars; i++)
                        {
                                startpt[i] = 4.0*erand48(buffer)-4.0;
                        }

                        local_jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax);
#if DEBUG 
		printf("\n\n\nHOOKE %d USED %d ITERATIONS, AND RETURNED\n", trial, local_jj);
		for (i = 0; i < nvars; i++)
			printf("x[%3d] = %15.7le \n", i, endpt[i]);
#endif

                        local_fx = f(endpt, nvars);
#if DEBUG
		printf("f(x) = %15.7le\n", local_fx);
#endif
                        if (local_fx < best_fx)
                        {
                                        #pragma omp atomic write
                                        best_trial = trial;

                                        #pragma omp atomic write
                                        best_jj = local_jj;

                                        #pragma omp atomic write
                                        best_fx = local_fx;

                                        #pragma omp critical
                                        {
                                                for (i = 0; i < nvars; i++)
                                                        best_pt[i] = endpt[i];
                                        }
                        }
                }

                // add thread-local funevals to node_funevals, which stores the total number of funevals for each node/MPI process
		#pragma omp atomic
		node_funevals += funevals;
        }
        /* Similarly, to the barrier above, we need this barrier to make sure that all nodes
         * are done with their computations, in order to begin the message passing part of the
         * program, which leads to a reduction, and in order to get the end time.
         */
        
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)   // node 0, will be responsible for doing the reduction and comparison of the individual results
        {
                double tmp = 0.0;
                for(int i=1; i<nprocs; i++)
                {
                        MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if(tmp < best_fx)     //if better than best_fx of root rank
                        {
                                best_fx = tmp;
                                MPI_Recv(best_pt, nvars, MPI_DOUBLE, i, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                MPI_Recv(&best_jj, 1, MPI_INT, i, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                MPI_Recv(&best_trial, 1, MPI_INT, i, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                }
        }
        else if(rank != 0)
        {
                MPI_Send(&best_fx, 1, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
                MPI_Send(best_pt, nvars, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
                MPI_Send(&best_jj, 1, MPI_INT, 0, 22, MPI_COMM_WORLD);
                MPI_Send(&best_trial, 1, MPI_INT, 0, 22, MPI_COMM_WORLD);
        }

        /* Gather the node_funevals (i.e each node's funevals) and reduce them using the sum operator,
         * into root rank's node_funevals variable
         */
        MPI_Reduce(rank==0? MPI_IN_PLACE: &node_funevals, &node_funevals, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank == 0)
                t1 = MPI_Wtime();
        
        if(rank == 0)   // rank 0, prints the results
	{
                printf("\n\nFINAL RESULTS:\n");
                printf("Elapsed time = %.3lf s\n", t1-t0);
                printf("Total number of trials = %d\n", ntrials);
                printf("Total number of function evaluations = %ld\n", node_funevals);
                printf("Best result at trial %d used %d iterations, and returned\n", best_trial, best_jj);
                for (i = 0; i < nvars; i++) {
                        printf("x[%3d] = %15.7le \n", i, best_pt[i]);
                }
                printf("f(x) = \t %15.7le\n", best_fx);
        }

        MPI_Finalize(); // Finalize/destroy the MPI environment
        
	return 0;
}