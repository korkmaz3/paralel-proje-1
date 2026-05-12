#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size, N;
    double *A, *B, *C, *sub_A, *sub_C;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        FILE *fA = fopen("a.txt", "r");
        FILE *fB = fopen("b.txt", "r");
        if (fA == NULL || fB == NULL) {
            printf("Hata: a.txt veya b.txt dosyasi bulunamadi veya acilamadi!\n");
            MPI_Abort(MPI_COMM_WORLD, 1); // Programı güvenli şekilde sonlandır
        }
        fscanf(fA, "%d", &N);
        fscanf(fB, "%d", &N);

        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));

        for (int i = 0; i < N * N; i++) fscanf(fA, "%lf", &A[i]);
        for (int i = 0; i < N * N; i++) fscanf(fB, "%lf", &B[i]);
        
        fclose(fA); fclose(fB);
        start_time = MPI_Wtime();
    }

    // N değerini herkese gönder
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size;
    sub_A = (double*)malloc(rows_per_proc * N * sizeof(double));
    sub_C = (double*)malloc(rows_per_proc * N * sizeof(double));
    if (rank != 0) B = (double*)malloc(N * N * sizeof(double));

    // B'yi herkese, A'nın satırlarını ise parça parça dağıt
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE, sub_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Çarpma işlemi
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            sub_C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                sub_C[i * N + j] += sub_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Sonuçları topla
    MPI_Gather(sub_C, rows_per_proc * N, MPI_DOUBLE, C, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("N=%d, Proc=%d, Süre=%f sn\n", N, size, end_time - start_time);
        free(A); free(B); free(C);
    }
    
    free(sub_A); free(sub_C);
    if (rank != 0) free(B);
    
    MPI_Finalize();
    return 0;
}
