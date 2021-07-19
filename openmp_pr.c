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

/* funevals is the number of times that the Rosenbrock objective function, has been called
 * to compute its value at a given point in space. This global variable, was used in the 
 * serial version of the program, without any problems occuring. However, when using a 
 * multithreaded programming paradigm, since all the threads share this variable,
 * it results in a race condition. Thus, the variable needs to be protected from concurrent
 * updates. We could wrap it in a critical region, like #pragma omp critical, or we could
 * make the updates atomic operations, using #omp atomic. However, by examining the code below
 * we can see that, when a thread creates a random vector and calls hooke(), hooke()
 * in turn calls function f() explicitly, and implicitly when best_nearby() is called. So, each 
 * thread must have its own copy of funevals and then each thread must add its funevals value,
 * using an atomic operation, to the total_funevals global variable.
 * 
 * If we used the **private** clause in order to give each thread its own private copy of the funevals global variable,
 * the following problem would be encountered:
 * 
 * Private variables are local to a region and are, most of the times, stack-allocated. Thus, when a thread calls a 
 * function which in turn increments the funevals variable, it is not incrementing the thread-private copy, but 
 * the ORIGINAL GLOBAL variable. Even if we used a reduction clause, it would still result in a race condition towards
 * the original global value, because as stated, when a thread calls a function and leaves the data environment where the
 * private copy of the variable, lives , that function accesses the global shared variable and not the thread-local one.
 * Such behavior is expected since private variables are stack-allocated.
 * 
 * On the other hand, if each thread has a **threadprivate** copy, then that copy is heap-allocated, so the thread-local
 * variable persists across regions. When a thread calls a function whose purpose is to alter the value of the global value,
 * the variable that is being changed is the threadprivate copy and NOT the original global variable. The only thread, that
 * is acting upon the original value is the main thread (the main thread is storage-associated with the original global variable,
 * while the other threads use their threadprivate copy,they are not storage-associated with the original global variable).
 * 
 * Therefore, we are using a threadprivate copy of funevals for each thread, in order to increment its own local copy of the variable,
 * when the thread calls functions like hooke(), that call f(), which in turn increments funevals.
 * This way when each thread has completed its work, it uses an atomic operation to add its funevals value to the
 * total_funevals global variable, which is shared among all threads.
 * The total_funevals variable, stores the total number of times the Rosenbrock objective function f(), was called.
 */                                                           
unsigned long funevals = 0;
#pragma omp threadprivate(funevals)
unsigned long total_funevals = 0;

/* Rosenbrocks classic parabolic valley ("banana") function .
 * 
 * No need to parallelize the internal for loop, as the number of iterations is small, but
 * even if it wasn't, there is no reason to parallelize this loop, as it would result in errors.
 * This is a task that each thread must do for itself.
 */                                                           
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

/* given a point, look for a better one nearby, one coord at a time
 *
 * Similarly, this doesn't need to be parallelized. Each thread calls this function
 * using a point in the 16-dimensional space, and tries to find a better nearby point, thus
 * trying to exploit the search space, rather than exploring it.
 */
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

	t0 = omp_get_wtime();

	/* Each thread will create its own random vector of size 16 (a point in the 16-dimensional space),
	 * which is stored in the startpt array. That point is used as a starting guess for the global minimum.
	 * The resulting best point, that hooke is going to find, will be stored in the endpt array.
	 * Each thread has its own local_fx variable, where it stores its minimum objective function value.
	 * The starting point is randomly created using numbers in the interval [-4,4). The random numbers
	 * are created using erand48() which is thread safe, and not using drand48() which internally uses static structs.
	 * Each thread has its own buffer to pass to erand48(), where its third element is based on the thread ID.
	 *
	 * After creating its random startpt, each thread calls hooke(), which will search in the nearby region
	 * to find a better point,  which is returned in the third argument, endpt().
	 * Upon returning from hooke(), each thread calls the Rosenbrock objective function to compute its value at the endpt,
	 * and compare it with the current total best_fx value.
	 * If it is less than the current best_fx, global variables like the number of iterations of the hooke function (jj),
	 * the best trial , etc, are updated using atomic write operations, which are faster than
	 * critical sections lock-based concurrency protection mechanisms. Finally, endpt is copied into best_pt, which is shared
	 * among all threads.
	 * 
	 * Finally, the thread-specific value of funevals, is added to the total_funevals. 
	 */ 

	/* Inside the parallel region we use the dynamic loop scheduling policy, with the
	 * default chunksize which is equal to 1. (Increasing the chunksize resulted in higher
	 * execution times)
	 * 
	 * After experimenting, it was observed that the dynamic policy was faster
	 * than both the static and guided policies.
	 * Specifically, when comparing individual runtimes, dynamic offered a speedup
	 * of 0.1 to 0.3 seconds, thus giving an average speedup of ~0.2 seconds.
	 * 
	 * Such a result was expected since, the execution time of each iteration is not
	 * constant, as a call to hooke might take more time to finish depending on the startpt
	 * and on the success of the best_nearby function to exploit the search space, near the given point.
	 * 
	 * Therefore, since each iteration has a different computational cost, it's better to use
	 * the **dynamic** loop scheduling policy, over the guided policy which is best applied when 
	 * poor load balancing occurs during the final iterations/stages of computation.
	 * 
	 * The dynamic policy obviously outweighs the static one where, each thread is only once given a chunk of iterations.
	 * This way, if a thread was assigned a chunk of iterations that require less time
	 * than the other threads' chunks, the first thread will have to wait for the other ones to finish
	 * thus leaving its computational power, unused, which in turn results in the higher execution time observed.
	 */

	/* The variables listed in the shared clause, are not the only ones that are being shared among all threads.
	 * The following ones, are listed to improve understanding, of the following code */
        #pragma omp parallel shared(rho, epsilon, itermax, total_funevals) 
        {
                double startpt[MAXVARS], endpt[MAXVARS];
                double local_fx = 0.0;
                int local_jj = 0;
                unsigned short buffer[3];
                buffer[0] = 0;
                buffer[1] = 0;
                buffer[2] = time(NULL) + omp_get_thread_num();

                #pragma omp for schedule(dynamic) nowait
                for (trial = 0; trial < ntrials; trial++)
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