#include<stdio.h>
#include<math.h>
#include <stdlib.h>

double norm(double x[], int n){
	double sqsum = 0;

	//parallelizable with a reduction clause
	for(int i =0; i<n; i++){
		sqsum += x[i]*x[i];
	}
	return sqrt(sqsum);
}

void print_matrix(int nrowA, int ncolA, double A[][ncolA]){
	/* Print matrix A*/

	//Not meant for parallelization
	for(int i=0;i<nrowA; i++){
		for (int j=0; j<ncolA; j++){
			//printf("%f ", *(A + i*ncolA + j));
			printf("%f ", A[i][j]);
		}
		printf("\n");
	}
}

void random_init(int nrowA, int ncolA, double A[][ncolA]){
	//Completely parallelizable
	for (int i=0; i<nrowA; i++){
		for (int j=0; j<ncolA; j++){
			//*(A + i*ncolA + j) = (i+1)*10 + j+1;
			A[i][j] = (i)*ncolA + j;
		}
	}
}

void transpose( int nrowA, int ncolA, double A[][ncolA], double A_transpose[][nrowA]){

	// Completely parallelizable
	for (int i=0; i<nrowA; i++)
		for (int j=0; j<ncolA; j++)
			A_transpose[j][i] = A[i][j];
}

void matmul(int nrowA, int ncolA, double A[][ncolA], int nrowB, int ncolB, double B[][ncolB], double C[][ncolB]){
	/*Matrix multiplication*/
	if (ncolA != nrowB){
		// raise error
		printf("Matrix dimensions do not match for multiplication! \n");
	}

	//Parallelizable
	for (int i = 0; i<nrowA; i++)
		for(int j=0; j<ncolB; j++)
			C[i][j] = 0;

	// Parallelizable 
	for (int i=0; i<nrowA; i++){
		for (int j=0; j<ncolB; j++){
			// require reduction
			for (int k=0; k<nrowB; k++){
				C[i][j] = C[i][j] + A[i][k]*B[k][j];
			}
		}
	}
}

void copy(int nrowA, int ncolA, double A[][ncolA], double B[][ncolA]){
	/*Copy values of A to B. B should be of same size as A*/

	//Completely parallelizable
	for (int i=0;i<nrowA; i++)
		for(int j=0; j<ncolA; j++)
			B[i][j] = A[i][j];
}

void Hadamard(int nrow, int ncol, double A[][ncol], double B[][ncol], double C[][ncol]){
	/* nrow - number of rows in A and B
	 ncol - number of columns in A and B
	 A is a two dimensional matrix
	 B - second two dimensional matrix
	 C - resultant matrix*/

	// Completely parallelizable
	for(int i=0; i<nrow; i++)
		for(int j=0; j<ncol; j++)
			C[i][j] = A[i][j] * B[i][j];
}

int main(int argc, char* argv[]){
	printf("Hello world!\n");
	// Check if norm works
	double x[] = {2.1,3.2,1.2,3.2};
	double y= norm(x, 2);
	printf("%f\n", y);

	// Hadamard Product
	int nrow = 2; 
	int ncol = 3;
	//double* A = (double*) malloc (nrow*ncol*sizeof( double ));
	double A[nrow][ncol], B[nrow][ncol], C[nrow][ncol];
	random_init(nrow, ncol, A);random_init(nrow, ncol, B);
	Hadamard(nrow, ncol, A,B,C);

	double C_T[ncol][nrow];
	transpose(nrow, ncol, C, C_T);
	print_matrix(nrow, ncol, B);
	print_matrix(nrow, ncol, C);
	print_matrix(ncol, nrow, C_T);
	double Ctc[ncol][ncol];
	matmul(ncol, nrow, C_T, nrow, ncol, C, Ctc);
	printf("\n");
	print_matrix(ncol, ncol, Ctc);

	return 0;
}