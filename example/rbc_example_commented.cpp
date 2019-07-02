#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "mpi.h"
#include "stdio.h"
#include "time.h"
#include "RBC.hpp"

//Estructura para el indice de un elemento particular
struct Ind2D {
  Ind2D(int _i=0, int _j=0) {
    k = _i;
    j = _j;
  }
  int k;
  int j;
};

typedef float** Matrix;
void SerialJacobi(Matrix mat, const int n, const float precision);
void JacobiParalelo(Matrix mat, const int n, const float precision);
void imprimirMatriz(Matrix mat, const int i);



//Función para encontrar el índice del mayor elemento fuera de la diagonal.
Ind2D getAbsIndiceMax(Matrix m, const int n) {
  float max = m[1][0];
  Ind2D max_ind(1, 0);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      if (j != k && abs(m[j][k]) > abs(max)) {
        max = m[j][k];
        max_ind.j = j; //Actualizar el indice del maximo
        max_ind.k = k;
      }
    }
  }
  return max_ind;
}


Matrix inicializarMatriz(const int n,long double v[]){
	int k=0;
  Matrix matrix = new float*[n];
  for (int i = 0; i < n; i++) {
    matrix[i] = new float[n];
  }
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      matrix[i][j] = v[k];
	k++; }
  }
  return matrix;
}

//Calcular la Norma de la matriz
float getNormaMatriz(Matrix m, const int n) {
  float norm = 0;
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      if (j != k) {
        norm += m[j][k] * m[j][k];
      }
    }
  }
  return std::sqrt(norm);
}

void imprimirMatriz(Matrix m, const int n) {
  for ( int i = 0; i < n; i++) {
    printf("Fila %d\n",i+1);
    for(int j=0;j<n;j++)
    {
      printf("%f\t ", m[i][j]);
    }
    printf("\n\n");
  }
}

//Función para obtener los ranks impares y pares de threads
int** getParImparRanks(int world_ranksize) {
  int** ranks = new int*[2];
  ranks[0] = new int[world_ranksize / 2];
  ranks[1] = new int[world_ranksize / 2];
  for (int i = 1; i < world_ranksize; ++i) {
      if (i % 2 == 1) {
        ranks[0][(i + 1) / 2 - 1] = i;

      } else {
        ranks[1][(i / 2) - 1] = i;
      }
  }
  //ranks[0] impares
  //ranks[1] pares
  return ranks;
}

/* Cálculos básicos del algoritmo paralelo de Jacobi para calcular autovalores de una matriz simétrica
 * El número mínimo de procesos para esta implementación es 3:
 * EL proceso 0 calcula los valores de la matriz de rotación y acumula datos de otros procesos
 * Los procesos 1 y 2 calculan nuevos elementos diagonales.
 * Los procesos 3 y 4 calculan nuevos valores de cadenas j y k
 * Otros procesos también calculan nuevos valores de cadenas j y k.*/

void RotacionJacobiParalela(Matrix m, int ind_j, int ind_k, const int n) {
  int rank, world_ranksize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  MPI_Comm_size(MPI_COMM_WORLD, &world_ranksize);

  MPI_Comm comm = MPI_COMM_WORLD; //Comunicador Padre
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  float c, s;
  int j, k;
  float* row_j = new float[n];
  float* row_k = new float[n];

  if (rank == 0) {
    j = ind_j;
    k = ind_k;
    //  Cálculo de los ángulos de la matriz de rotación.
    if (m[j][j] == m[k][k]) {
      c = cos(M_PI / 4);
      s = sin(M_PI / 4);
    }
    else {
      float tau = (m[j][j] - m[k][k]) / (2 * m[j][k]);
      float t = ((tau > 0) ? 1 : -1) / (abs(tau) + sqrt(1 + tau * tau));
      c = 1 / sqrt(1 + t * t);
      s = c * t;
    }
    // Copia las líneas j y k en los buffers apropiados para enviarlos a otros threads.
    for (int i = 0; i < n; ++i) {
      row_j[i] = m[j][i];
      row_k[i] = m[k][i];
    }

  }

  // Enviamos datos a todos los threads
  RBC::Bcast(&j, 1, MPI_INT, 0, rcomm);  //  linea numero j
  RBC::Bcast(&k, 1, MPI_INT, 0, rcomm);  //  linea numero k
  RBC::Bcast(&c, 1, MPI_FLOAT, 0, rcomm);  // cos(theta)
  RBC::Bcast(&s, 1, MPI_FLOAT, 0, rcomm);  // sin(theta)
  RBC::Bcast(row_j, n, MPI_FLOAT, 0, rcomm);  // linea j
  RBC::Bcast(row_k, n, MPI_FLOAT, 0, rcomm);  // linea k
  float m_jj, m_kk;
  
  /* 
  Creamos dos grupos de procesos: uno para recalcular la cadena j, el otro para convertir la cadena k
  El grupo de cadenas k consta de procesos impares. (mayor o igual a 3)
  El grupo de cadenas j consiste en procesos pares. (mayor o igual a 4)
  */
  //int** odd_even_ranks = getParImparRanks(world_ranksize); //get the thread numbers

  const int group_k_size = (world_ranksize - 1) / 2;
  const int group_j_size = group_k_size;
  //MPI_Group_incl(world_group, group_j_size, odd_even_ranks[0], &row_j_group);
  //MPI_Group_incl(world_group, group_k_size, odd_even_ranks[1], &row_k_group);
  // Creamos comunicadores basados en grupos.
  //MPI_Comm row_j_comm;
  //MPI_Comm row_k_comm;
  /*
  MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm* newcomm)
   */
  //MPI_Comm_create_group(MPI_COMM_WORLD, row_j_group, 0, &row_j_comm);
  //MPI_Comm_create_group(MPI_COMM_WORLD, row_k_group, 0, &row_k_comm);

  // creación de Sub-group en tiempo constante sin comunicación
  RBC::Comm row_j_comm; //impares
  RBC::Comm row_k_comm; //pares
  int stride2 = 2;
  RBC::Comm_create_group(rcomm, &row_j_comm, 1, group_j_size, stride2); //impares
  RBC::Comm_create_group(rcomm, &row_k_comm, 0, group_k_size, stride2); //pares

  int row_j_rank = -1;
  int row_j_size = -1;
  float* row_j_new = new float[n];
  //  check the existence of the communicator
  if (rank%2 == 1) { // MPI_COMM_NULL != row_j_comm

    row_j_size = row_j_comm.getSize();
    row_j_rank = row_j_comm.getRank();

    //  the portion of the row that one stream recounts from the row_j_comm group
    int size = n / row_j_size;
    //  Allocate memory for buffers
    float* row_j_part = new float[size];
    float* row_k_part = new float[size];
    float* row_j_new_part = new float[size];
    //  We divide k and j rows between processes of row_j_comm group
    RBC::Scatter(row_j, size, MPI_FLOAT, row_j_part, size, MPI_FLOAT, 0, row_j_comm);
    RBC::Scatter(row_k, size, MPI_FLOAT, row_k_part, size, MPI_FLOAT, 0, row_j_comm);
    //  Recount part of the line
    for (int i = 0; i < size; ++i) {
        row_j_new_part[i] = c * row_j_part[i] + s * row_k_part[i];
    }
    // We assemble a new line from the parts in the 0 process in relation to the row_j_comm communicator
    // (3 - with respect to MPI_COMM_WORLD)
    RBC::Gather(row_j_new_part, size, MPI_FLOAT, row_j_new, size, MPI_FLOAT, 0, row_j_comm);
    if (row_j_rank == 0) {
      // We send a new line to the 0 process (in relation to MPI_COMM_WORLD)
      RBC::Send(row_j_new, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    // Liberar la memoria, group and communicator
    delete[] row_j_new_part;
    delete[] row_k_part;
    delete[] row_j_part;
    RBC::Comm_free(&row_j_comm);
  }

  int row_k_rank = -1;
  int row_k_size = -1;
  float* row_k_new = new float[n];
  if (rank%2 == 0){ //(MPI_COMM_NULL != row_k_comm) {

    row_k_size = row_k_comm.getSize();
    row_k_rank = row_k_comm.getRank();

    int size = n / row_k_size;
    float* row_j_part = new float[size];
    float* row_k_part = new float[size];
    float* row_k_new_part = new float[size];
    RBC::Scatter(row_j, size, MPI_FLOAT, row_j_part, size, MPI_FLOAT, 0, row_k_comm);
    RBC::Scatter(row_k, size, MPI_FLOAT, row_k_part, size, MPI_FLOAT, 0, row_k_comm);
    for (int i = 0; i < size; ++i) {
        row_k_new_part[i] = s * row_j_part[i] - c * row_k_part[i];
    }
    RBC::Gather(row_k_new_part, size, MPI_FLOAT, row_k_new, size, MPI_FLOAT, 0, row_k_comm);
    if (row_k_rank == 0) {
      RBC::Send(row_k_new, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    delete[] row_k_new_part;
    delete[] row_k_part;
    delete[] row_j_part;
    Comm_free(&row_k_comm);
  }
  //  Modifying the matrix in the main process
  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
     
    RBC::Recv(row_j_new, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // k line recalculated
    RBC::Recv(row_k_new, n, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Replace the values ​​of the original matrix
    m[j][k] = (c * c - s * s) * row_j[k] + s * c * (row_k[k] - row_j[j]);
    m[k][j] = m[j][k];
    
    m[j][j] = c * c * row_j[j] + 2 * s * c * row_j[k] + s * s * row_k[k];
    m[k][k] = s * s * row_j[j] - 2 * s * c * row_j[k] + c * c * row_k[k];;
    for (int i = 0; i < n; ++i) {
      if (i != j && i != k) {
        m[j][i] = row_j_new[i];
        m[k][i] = row_k_new[i];
        m[i][j] = m[j][i];
        m[i][k] = m[k][i];
      }
    }
  }
  //Liberar la memoria
  delete[] row_k_new;
  delete[] row_j_new;
  delete[] odd_even_ranks[1];
  delete[] odd_even_ranks[0];
  delete[] odd_even_ranks;
  delete[] row_k;
  delete[] row_j;
}

// Input: MATRIZ, orden de la matriz, precision
void JacobiParalelo(Matrix mat, const int n, const float precision) {
  Ind2D indiceMax;
  float elapsed_time = 0;
  indiceMax = getAbsIndiceMax(mat, n);
  float norm = getNormaMatriz(mat, n);;
  float tol; int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    norm = getNormaMatriz(mat, n);
    tol = precision * norm;
    printf("precisión = %f, norma = %f, tol = %f\n",precision, norm, tol);
  }
  //Enviar norma y tol
  MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  while (norm > tol) {
    elapsed_time -= MPI_Wtime();
    RotacionJacobiParalela(mat, indiceMax.j, indiceMax.k, n);
    if (rank == 0) {
      norm = getNormaMatriz(mat, n);
    }
    elapsed_time += MPI_Wtime();
    MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    indiceMax = getAbsIndiceMax(mat, n);
  }
  
}

int main(int argc, char** argv){
  int n = std::atoi(argv[1]); // ./rbc_jacobi 58, 58 es la dimensión
  float start=0, end=0,a;
  double elapsed_time = 0;
  long double v[n*n];
  int i=0,j=0;
  int my_rank;

  std::fstream myfile("dataset.txt", std::ios_base::in);
     
  while (myfile >> a) //Leer dataset
  {    
    v[i]=a;
    i++;
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  elapsed_time -= MPI_Wtime();
  Matrix m = inicializarMatriz(n,v);

  if (my_rank == 0) //debug
    imprimirMatriz(m, n); //entra la matriz y el orden de la matriz
    	 
  JacobiParalelo(m, n, 1e-5);
  elapsed_time += MPI_Wtime();

  if (my_rank == 0) {
	  printf("\nLos autovalores de la matriz son:-\n\n");
    for (int i = 0; i < n; ++i) {
      printf("\t X[%d] \t=\t %f \n",i, m[i][i]);
    }
    printf("\n\t Dimensión de la matriz = %i\n\t Tiempo total = %fsecs\n", n, elapsed_time);
   
  }
  MPI_Finalize();
 
  return 0;

}

