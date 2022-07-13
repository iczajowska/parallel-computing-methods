#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

double synchronized_send(unsigned long long iterations){
    double time_start, time_end;
    unsigned long long i;
    
    MPI_Barrier(MPI_COMM_WORLD);

    time_start = MPI_Wtime();
    for (i= 0; i < iterations; i++) {
        MPI_Ssend(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    time_end = MPI_Wtime();

    return (time_end - time_start) / (double) iterations;
}

double buffered_send(unsigned long long iterations) {
    double time_start, time_end;
    unsigned long long i;
    long msg_size = 1;
    char msg = '1';
    int att_buffer_size;

    MPI_Pack_size(msg_size, MPI_CHAR, MPI_COMM_WORLD, &att_buffer_size);
    att_buffer_size = MPI_BSEND_OVERHEAD + sizeof(char);
    char* buffer = (char*)malloc(att_buffer_size);
    MPI_Buffer_attach(buffer, att_buffer_size);

    MPI_Barrier(MPI_COMM_WORLD);

    time_start = MPI_Wtime();
    for (i= 0; i < iterations; i++) {
        MPI_Bsend(&msg, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    time_end = MPI_Wtime();

    MPI_Buffer_detach(&buffer, &att_buffer_size);
    free(buffer);

    return (time_end - time_start) / (double) iterations;
}

void synchronized_receiver(unsigned long long iterations) {
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long long i;
    for (i = 0; i < iterations; i++) {
        MPI_Recv(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void buffered_reciver(unsigned long long iterations) {
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long long i;
    char msg;
    for (i = 0; i < iterations; i++) {
        MPI_Recv(&msg, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) { 
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(argc < 3){
        // method: 0 - MPI_Ssend, 1 - MPI_Bsend
        fprintf(stderr, "expected: method, number of iterations\n");
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }

    if(world_size !=2) {
        fprintf(stderr, "expected two processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }

    double iter;
    int method = atoi(argv[1]);
    unsigned long long iterations;
    sscanf(argv[2],"%lf",&iter);
    iterations = iter;

    if (world_rank == 0) {
        if (method == 0){
            printf("MPI_Ssend latency: %f\n",synchronized_send(iterations));
        } else {
            printf("MPI_BSend latency: %f\n",buffered_send(iterations));
        }
    } else {
        if(method == 0){
            synchronized_receiver(iterations);
        } else {
            buffered_reciver(iterations);
        }
    }

    MPI_Finalize();
    return 0;
}