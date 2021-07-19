#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>


#define MAXVARS		(250)	/* max # of variables	     */
#define RHO_BEGIN	(0.5)	/* stepsize geometric shrink */
#define EPSMIN		(1E-6)	/* ending value of stepsize  */
#define IMAX		(5000)	/* max # of iterations	     */

/* global variables */

/* In this version, we are using parallelization using OpenMP Tasks.
 * Each thread will execute tasks from the OMP Runtime Environment Task Queue, so we again need each thread
 * to have its own private copy of the funevals variable.
 * 
 * If we allowed each task to have its own private copy of the funevals variable, when a task calls hooke, which in turn calls f(), 
 * the global shared funevals variable would be incremented and not the task-specific one. This results in a race condition.
 * 
 * Likewise, we cannot give a private copy of the funevals variable to each thread.
 * 
 * For the reasons mentioned above, again the private clause is not a proper choice, since when a task running on a particular thread,
 * calls a function (i.e. hooke) that will in turn call f() the variable that will be incremented is the shared global funevals,
 * and not the thread-specific one.
 * 
 * So again, each thread must have its own **threadprivate** copy of the funevals variable, which is persistent across regions.
 * This way, when a task is running on a thread, every call that the task makes will result in the thread-specific copy of the funevals
 * variable, to be incremented and not the global one.
 * Notice that we don't care about the number of function evaluations that each task did. We only care about the total number of times f()
 * was called.
 * So it doesn't matter how many or which tasks are being executed on each thread, as long as each task is only incrementing its executing
 * thread's copy of funevals.
 * 
 * Thus, every thread uses an atomic operation to add its value of funevals to the shared total_funevals, in which we will store the total
 * number of function evaluations of the Rosenbrock objective function.
 * 
 * (see multistart_hooke_omp.c for more on private vs threadprivate variables)
 */ 

unsigned long funevals = 0;
#pragma omp threadprivate(funevals)

unsigned long total_funevals = 0;


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


double get_wtime(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

int main(int argc, char *argv[])
{
	/* uncomment if the runtime environment doesn't run the program with the max number of hardware threads available
	*  which is the default behavior
	*/
	// omp_set_dynamic(0);
	// omp_set_num_threads(omp_get_num_procs());

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

	ntrials = 128 * 1024;	/* number of trials */
	nvars = 16;		/* number of variables (problem dimension) */
	srand48(time(0));

        unsigned long ntasks = 128 * 2;
        unsigned long k;
        long tseed = time(NULL);

	t0 = omp_get_wtime();

	/* Create a parallel region, where one thread will be the task-creator, while others wait for tasks
	 * to be put in the Runtime Environment's Task Queue.
	 * The code inside each task, is almost identical to the one in the multistart_hooke_omp.c file.
	 * The only difference is that, here every task has a fixed number of steps to execute, before terminating.
	 * In order to achieve a similar behavior in the threads-only version, we used a worksharing for construct.
	 * As with the threads-only version, each task will create a random startpt, which will be used a starting guess
	 * for the minimum of the Rosenbrock function. Then, hooke() is called and the best point near the initial one,
	 * is returned in the endpt variable. Then we check if this point results in a value less than the current best_fx,
	 * and if so, we update the variables storing information regarding the best current point/value.
	 */

        #pragma omp parallel shared(total_funevals)
        {
                #pragma omp single nowait
                {
                        for(k = 0; k < ntasks; k++)
                        {
                                #pragma omp task firstprivate(k) shared(rho, epsilon, itermax, best_trial, best_pt, best_jj,nvars)
                                {
                                        unsigned short buff[3];
                                        buff[0] = 0;
                                        buff[1] = 0;
                                        buff[2] = tseed + k;

                                        double startpt[MAXVARS], endpt[MAXVARS];
                                        int local_jj = 0;
                                        double local_fx = 0;
                                        long nsteps = ntrials / ntasks;

                                        for(long l = 0; l < nsteps; l++)
                                        {
                                                for(int e = 0; e < nvars; e++)
                                                        startpt[e] = 4.0 * erand48(buff) - 4.0;
                                        
                                                local_jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax);
                                                long curr_trial = k * ntasks + l;
#if DEBUG 
                                                printf("\n\n\nHOOKE %d USED %d ITERATIONS, AND RETURNED\n", curr_trial, local_jj);
                                                for (i = 0; i < nvars; i++)
                                                        printf("x[%3d] = %15.7le \n", i, endpt[i]);
#endif
                                        
                                                local_fx = f(endpt, nvars);
#if DEBUG
                                		printf("f(x) = %15.7le\n", local_fx);
#endif

                                                if(local_fx < best_fx)
                                                {
                                                        #pragma omp atomic write
                                                        best_fx = local_fx;

                                                        #pragma omp atomic write
                                                        best_jj = local_jj;

                                                        #pragma omp atomic write
                                                        best_trial = curr_trial;

                                                        #pragma omp critical
                                                        {
                                                                for(int i=0; i< nvars; i++)
                                                                        best_pt[i] = endpt[i];
                                                        }
                                                }
                                        }
                                }
                        }
                }

		/* A barrier is needed at this point, in order to synchronize all threads.
		 * This is needed because, all threads (except the task-maker), will skip the single clause
		 * (due to the nowait clause being used), and reach this point. If we didn't set a barrier, 
		 * some threads, could execute the atomic operation of updating the total_funevals variable 
		 * by adding their value of the funevals threadprivate copy. This would result in erroneous 
		 * behavior, since if a thread hasn't executed any task, then its copy of the funevals variable,
		 * would have a value of 0. So we use the barrier in order to force all threads, to update the 
		 * total_funevals variable, only AFTER they have finished executing all the available tasks
		 * and the task queue is empty.
		 */
		
		#pragma omp barrier

		#pragma omp atomic
		total_funevals += funevals;		
        }
	
	t1 = omp_get_wtime();

	printf("\n\nFINAL RESULTS:\n");
	printf("Elapsed time = %.3lf s\n", t1-t0);
	printf("Total number of trials = %d\n", ntrials);
	printf("Total number of function evaluations = %ld\n", total_funevals);
	printf("Best result at trial %d used %d iterations, and returned\n", best_trial, best_jj);
	for (i = 0; i < nvars; i++) {
		printf("x[%3d] = %15.7le \n", i, best_pt[i]);
	}
	printf("f(x) = \t %15.7le\n", best_fx);

	return 0;
}