#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>


int L = 4;
int F = 2;

void print_mat(int s1, int s2,double mat[][s2]){
	for(int i=0;i<s1;i++){
		for(int f=0;f<s2;f++){
			printf("%f ",mat[i][f]);
		}
		printf("\n");
	}
}


double norm(double x[], int n){
	double sqsum = 0;

	//parallelizable with a reduction clause
	for(int i =0; i<n; i++){
		sqsum += x[i]*x[i];
	}
	return sqrt(sqsum);
}


void transpose( int nrowA, int ncolA, double A[][ncolA], double A_t[][nrowA]){

	// Completely parallelizable
	for (int i=0; i<nrowA; i++)
		for (int j=0; j<ncolA; j++)
			A_t[j][i] = A[i][j];
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
    
void Cholesky_Decomposition(int nrows, int ncols, double A[][ncols], double L[][ncols], int init_L)
{
	if (nrows!=ncols){
		// Raise error 
		printf("Input should be a square matrix!");
	}

	// initalize L
	// Loops are parallellizable
	if (init_L==0){
		for(int i =0; i< nrows; i++)
			for(int j = 0; j<ncols; j++)
				L[i][j] = 0;		
	}
 
    // Decomposing a matrix into Lower Triangular
    //Loop carried dependency and therefore not parallelizable along one of the loop
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j <= i; j++) {
            int sum = 0;
 
            if (j == i) // summation for diagnols
            {
                for (int k = 0; k < j; k++)
                    sum += L[j][k]*L[j][k];
                L[j][j] = sqrt(A[j][j] - sum);
            } else {
 
                // Evaluating L(i, j) using L(j, j)
                for (int k = 0; k < j; k++)
                    sum += (L[i][k] * L[j][k]);
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
}

void randvec(int size, double mat[]){
	srand(time(0));
	// Non-parallelizable
	for(int i=0; i<size; i++){
		mat[i] = (double)(rand() % 10000) / 10000.0;
	}
}

void randmat(int nrows, int ncols, double mat[][ncols]){
	srand(time(0));
	// Non-parallelizable
	for(int i=0; i<nrows; i++){
		for(int j=0; j<ncols; j++){
			mat[i][j] = (double)(rand() % 10000) / 10000.0;
	}}
}

void init_congruence_mat(double A[][F], double congruence){
	int i,j;

	//parallelizable loops
	for (i=0; i<F;i++){
		for (j = 0; j<F; j++){
			if (i==j){
				A[i][j] = 1.0;
			}else{
				A[i][j] = congruence;
			}
		}
	}
}

void gramschmith(int i, double U[][F]){
	for(int j = 0; j< i-1; j++){
			double component = 0.0;
			// dot product
			for (int k=0; k<L; k++){
				component += U[k][j]*U[k][i];
			}
			// This loop is parallelizable
			for(int k =0; k<L; k++){
				U[k][i] += -component*U[k][j];
			}
		}
}

void initU(double U[][F]){
	randmat(L, F, U);
	// Gram-schmidth orthogonalization
	for(int i=0; i<F; i++){
		gramschmith(i, U);

		//Normalization
		double norm = 0;
		for (int k =0; k<L; k++){
			norm = norm + pow(U[k][i],2);
		}

		while(norm<0.00001){
			printf("Need Reinitialization");
			for(int k=0;k<L;k++){
				U[k][i] = (double)(rand() % 10000) / 10000.0;
			}
			gramschmith(i,U);
			double norm = 0;
			for (int k =0; k<L; k++){
				norm += pow(U[k][i],2);
			}
		}

		//parallelizable
		for (int k =0; k<L; k++){
			U[k][i] = U[k][i]/norm;
		}
	}
	//Scaling randomly
	for(int i=0; i<F; i++){
		int randnum = rand() % 10; 
		for (int k=0; k<L; k++){
			U[k][i] = randnum * U[k][i];
		}
	}
}

void Loading_matrix(double loading_mat[][F]){
	// define Congruence matrix
	double A[F][F], R[F][F];
	double congruence = 0.6; // 0< colinearity < 1
	init_congruence_mat(A, congruence);
	Cholesky_Decomposition(F, F, A, R, 0);

	double U[L][F]; // column-orthogonal matrix
	initU(U);
	matmul(L, F, U, F, F, R, loading_mat);
}

int main(){
	double A[L][F], B[L][F], C[L][F];
	Loading_matrix(A);
	Loading_matrix(B);
	Loading_matrix(C);
	// Multiply A, B, C to get the final tensor
	double tensor[L][L][L];
	for(int i = 0; i<L; i++){
		for(int j=0; j<L; j++){
			for(int k=0; k<L; k++){
				for (int iter=0; iter<F; iter++){
					tensor[i][j][k] += A[i][iter]*B[j][iter]*C[k][iter];
				}
			}
		}
	}

	// print the matrix
	for(int i=0; i<L; i++)
		print_mat(L, L, tensor[i]);

	return 0;
}