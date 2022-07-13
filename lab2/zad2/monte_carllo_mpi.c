#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

unsigned long long monte_carlo(unsigned long long numberOfPoints) {
    unsigned long long i, circle;
    circle = 0;

    for(i = numberOfPoints; i > 0; i--){
        float randomX = (float)rand()/(float)RAND_MAX;
        float randomY = (float)rand()/(float)RAND_MAX;
        
        if((randomX * randomX) + (randomY * randomY) < 1) {
            circle++;
        }
    }

    return circle;
}

float get_pi(unsigned long long points, unsigned long long circle_points) {
    return (4.0*(float)circle_points/(float)points);
}

int main(int argc, char** argv) { 
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Generate different numbers 
    srand(time(NULL));

    if(argc < 2){
        fprintf(stderr, "expected: number of points\n");
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }

    double numberDouble, time_start, time;
    unsigned long long points, nodes_points, circle_points;

    sscanf(argv[1],"%lf",&numberDouble);
    points = numberDouble;
    nodes_points = points / world_size;

    // Convert data to easy read csv format
    // points,total_circle_points,processors,result,time\n
    if (world_rank == 0) {
        unsigned long long total_circle_points;

        MPI_Barrier(MPI_COMM_WORLD);
        time_start = MPI_Wtime();
        circle_points = monte_carlo(nodes_points);
        //Reduce - sendbuff, recvbuff, count, datatype, operation, root, comm
        MPI_Reduce(&circle_points, &total_circle_points, 1,
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        time = MPI_Wtime() - time_start;        
        
        printf("%llu,%llu,%d,%f,%f\n", points, total_circle_points, world_size, get_pi(points, total_circle_points), time);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        circle_points = monte_carlo(nodes_points);
        MPI_Reduce(&circle_points, NULL, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}