#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


// Convert data to easy read csv format
// message_size,time,throughput\n
void get_throughput(double time_start, double time_end, unsigned long long iterations, double problem_size) {
    double calculated_time = (time_end - time_start) / (double) iterations;
    double throughput = problem_size * 8 / (1024 * 1024) / calculated_time; 
    printf("%f,%f,%f\n", problem_size, calculated_time, throughput);
}

void synchronized_send(unsigned long long iterations, int message_size){
    double time_start, time_end;
    unsigned long long i;

    char* buffer = (char*)malloc(message_size * sizeof(char));
    for (i = 0; i < message_size; i++){
        buffer[i] = 'x';
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    time_start = MPI_Wtime();
    for (i= 0; i < iterations; i++) {
        MPI_Ssend(buffer, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    time_end = MPI_Wtime();

    get_throughput(time_start, time_end, iterations, message_size);
    
    free(buffer);
}

void buffered_send(unsigned long long iterations, int message_size) {
    double time_start, time_end;
    unsigned long long i;
    int msg_size = message_size;
    int att_buffer_size;

    char* msg = (char*)malloc(msg_size * sizeof(char));
    for (i = 0; i < msg_size; i++){
        msg[i] = 'x';
    }


    MPI_Pack_size(msg_size, MPI_CHAR, MPI_COMM_WORLD, &att_buffer_size);
    att_buffer_size += MPI_BSEND_OVERHEAD;
    char* buffer = (char*)malloc(att_buffer_size);
    MPI_Buffer_attach(buffer, att_buffer_size);

    MPI_Barrier(MPI_COMM_WORLD);

    time_start = MPI_Wtime();
    for (i= 0; i < iterations; i++) {
        MPI_Bsend(msg, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    time_end = MPI_Wtime();

    MPI_Buffer_detach(&buffer, &att_buffer_size);

    get_throughput(time_start, time_end, iterations, message_size);
    
    free(buffer);
    free(msg);
}

void synchronized_receiver(unsigned long long iterations, int message_size) {
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long long i;
    char* msg = (char*)malloc(message_size * sizeof(char));
    for (i = 0; i < iterations; i++) {
        MPI_Recv(msg, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    free(msg);
}

void buffered_receiver(unsigned long long iterations, int message_size) {
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long long i;
    char* msg = (char*)malloc(message_size * sizeof(char));
    for (i = 0; i < iterations; i++) {
        MPI_Recv(msg, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    free(msg);
}

int main(int argc, char** argv) { 
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(argc != 6){
        // method: 0 - MPI_Ssend, 1 - MPI_Bsend
        fprintf(stderr, "expected: method, number of iterations, message size [B], max message size [B], delta [B]\n");
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
    int message_size = atoi(argv[3]);
    int max_message_size = atoi(argv[4]);
    int delta = atoi(argv[5]);

    for (; message_size <= max_message_size; message_size += delta) {
        if (world_rank == 0) {
            if (method == 0){
                synchronized_send(iterations, message_size);
            } else {
                buffered_send(iterations, message_size);
            }
        } else {
            if(method == 0){
                synchronized_receiver(iterations, message_size);
            } else {
                buffered_receiver(iterations, message_size);
            }
        }
    }

    MPI_Finalize();
    return 0;
}