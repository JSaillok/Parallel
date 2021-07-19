#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define MAXVARS		(250)	/* max # of variables	     */
#define RHO_BEGIN	(0.5)	/* stepsize geometric shrink */
#define EPSMIN		(1E-6)	/* ending value of stepsize  */
#define IMAX		(5000)	/* max # of iterations	     */

/* global variables 
 *
 * No need to protect funevals, from concurrent writes, because in this version we are using
 * the MPI programming paradigm. We are no longer working with threads, but with several processes, instead.
 * So, there are no race conditions, since each process has its own memory address space.
 */
unsigned long funevals = 0;


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

/* Each process will test several randomly generated starting points
 * to find the one that minimizes the Rosenbrock objective function.
 * 
 * After a per-process reduction to get each node's optimal variables,
 * each rank sends its findings to the master rank (rank 0), which
 * checks whether the value it received from rank i (where i in [1,nprocs-1]), is 
 * better than the current total minimum value, and if so, it updates the corresponding variables.
 * 
 * Finally, there is a reduction on the funevals variable from each node, to get the total number
 * of times that the Rosenbrock function was called
 */

int main(int argc, char *argv[])
{       
        MPI_Init(&argc, &argv);         // initialize the MPI environment

        int rank, nprocs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	double startpt[MAXVARS], endpt[MAXVARS];
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

        /* needed for erand48 which is thread-safe/concurrent call safe    */
        unsigned short buff[3];
        buff[0] = 0;
        buff[1] = 0;
        buff[2] = tseed + rank;

        /* We need a barrier here, in order to be sure that all processes are done setting up their 
         * variables (i.e. their data environment), and are ready to proceed into their computational for-loop.
         * This is needed, in order to get the real starting time of the parallel computations.
         * 
         * If we didn't use a barrier here, we could end up getting a wrong starting time
         */
        MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0)
                t0 = MPI_Wtime();

        /* Each process (i.e node) will start with a loop-index equal to its rank, run up to ntrials-1,
         * by taking jumps of 'nprocs' size. For example, the root node (node 0) starts at iteration 0,
         * then proceeds to iteration 12, then 24 , etc.
         * This way, we use each node's rank and the total number of nodes, to effectively split the 
         * computational work of 'ntrials' steps, into (almost) equal sized chunks.
         */
	for(trial = rank; trial < ntrials; trial += nprocs)
	{
		/* starting guess for rosenbrock test function, search space in [-4, 4) */
		for (i = 0; i < nvars; i++)
		{
                        startpt[i] = 4.0 * erand48(buff) - 4.0;
		}

		jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax);
#if DEBUG 
		printf("\n\n\nHOOKE %d USED %d ITERATIONS, AND RETURNED\n", trial, jj);
		for (i = 0; i < nvars; i++)
			printf("x[%3d] = %15.7le \n", i, endpt[i]);
#endif

		fx = f(endpt, nvars);
#if DEBUG
		printf("f(x) = %15.7le\n", fx);
#endif
		if (fx < best_fx)
		{
			best_trial = trial;
			best_jj = jj;
			best_fx = fx;
			for (i = 0; i < nvars; i++)
				best_pt[i] = endpt[i];
		}
	}
        /* Similarly, to the barrier above, we need this barrier to make sure that all nodes
         * are done with their for loop, in order to begin the message passing part of the
         * program, which leads to a reduction, and in order to get the end time.
         * 
         * All ranks other than rank 0, send their results (i.e their best_fx, at which point it
         * occured (best_pt), etc,), to rank 0 who is responsible for comparing the received results
         * and the current best ones, in order to find the optimal ones.
         * 
         * In rank 0, if the value of the Rosenbrock function is better than our current best one,
         * update the current best value, and copy the received best_pt into the root rank's best_pt variable.
         * We act the same way for all the other variables associated with the global minimum value of the
         * Rosenbrock objective function.
         * 
         * We cannot simply use MPI_Reduce to place the results into the root rank, because other than
         * the best_fx, best_jj and best_trial optimal values (which are integers and a lower value is a better one),
         * we also need the best_pt, i.e. the point were the Rosenbrock function takes its global minimum value.
         * 
         * We cannot obtain the best_pt, using MPI_Reduce, because there is no comparison to be made, between the 
         * coordinates of each point. Therefore, we need to receive the results from each non-root rank, and check
         * if the received best_fx value is better than the current one, and if so, set best_pt equal to the sender
         * rank's, best_pt vector.
         * This results in an O(N) time complexity for the reduction operation.
         */
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)   // node 0, will be responsible for doing the reduction and comparison of the individual results
        {
		double tmp;
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
		/* After this for loop, the total minimum value of the Rosenbrock function, will be stored 
		 * in root rank's best_fx variable. Also, all the variables associated with a minimum, like
		 * best_jj, best_trial, best_pt that give the total minimum, will be stored into root rank's
		 * corresponding variables
		 */
        }

        else if(rank != 0)
        {
                MPI_Send(&best_fx, 1, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
                MPI_Send(best_pt, nvars, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
                MPI_Send(&best_jj, 1, MPI_INT, 0, 22, MPI_COMM_WORLD);
                MPI_Send(&best_trial, 1, MPI_INT, 0, 22, MPI_COMM_WORLD);
        }
        /* Rank 0 performs a reduction on the number of function evaluations of the objective function, summing
         * all individual nodes' values, into its own variable
         */
        MPI_Reduce(rank==0? MPI_IN_PLACE:&funevals, &funevals, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank == 0)
                t1 = MPI_Wtime();
        /* We could have let each rank have its own start and end time, and then make rank 0,
         * perform a reduction on them, to get the maximum wall clock execution time.
         * However, it was observed that the differences between each rank's time values, were
         * negligible and thus not worth the effort and overhead of doing a reduction operation.
         * So the printed elapsed time is the execution time of node 0, which is usually the higher one.
         */
        if(rank == 0)   // rank 0, prints the results
	{
                printf("\n\nFINAL RESULTS:\n");
                printf("Elapsed time = %.3lf s\n", t1-t0);
                printf("Total number of trials = %d\n", ntrials);
                printf("Total number of function evaluations = %ld\n", funevals);
                printf("Best result at trial %d used %d iterations, and returned\n", best_trial, best_jj);
                for (i = 0; i < nvars; i++) {
                        printf("x[%3d] = %15.7le \n", i, best_pt[i]);
                }
                printf("f(x) = \t %15.7le\n", best_fx);
        }

        MPI_Finalize(); // Finalize/destroy the MPI environment
        
	return 0;
}