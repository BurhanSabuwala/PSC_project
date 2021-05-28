#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>


int L = 50;
int F = 25;
double congruence = 0.17;
double PI = 3.14159;

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

double randn(double mu, double sigma){
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1.0 || W == 0.0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
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
			C[i][j] = 0.0;

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
            double sum = 0.0;
 
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
	// Non-parallelizable
	for(int i=0; i<size; i++){
		mat[i] = (double)(rand() % 10000) / 10000.0;
	}
}

void randmat(int nrows, int ncols, double mat[][ncols]){
	// Non-parallelizable
	for(int i=0; i<nrows; i++){
		for(int j=0; j<ncols; j++){
			mat[i][j] = (double)(rand() % 10000) / 10000.0;
	}}
}

double randtensor_normal(int nrows, int ncols, int nsli, double mat[][ncols][nsli]){
	// Non-parallelizable
	double sum = 0.0;
	for(int i=0; i<nrows; i++){
		for(int j=0; j<ncols; j++){
			for(int k=0; k<nsli; k++){
				double random_number = randn(0, 1);
				mat[i][j][k] = random_number;
				sum += pow(random_number,2);
			}
	}}
	return sqrt(sum);
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
	for(int j = 0; j< i; j++){
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
		}norm = sqrt(norm);

		while(norm<0.001){
			printf("Need Reinitialization");
			for(int k=0;k<L;k++){
				U[k][i] = (double)(rand() % 10000) / 10000.0;
			}
			gramschmith(i,U);
			double norm = 0;
			for (int k =0; k<L; k++){
				norm += pow(U[k][i],2);
			}norm =sqrt(norm);
		}

		// parallelizable
		for (int k =0; k<L; k++){
			U[k][i] = U[k][i]/norm;
		}
	}
	
}

void random_scaling_of_columns(double U[][F]){
	for(int i=0; i<F; i++){
		double randnum = 1 +(double)(rand() % 9); 
		for (int k=0; k<L; k++){
			U[k][i] = randnum * U[k][i];
		}
	}
}

double get_colinearity(int i, int j, double loading_mat[][F]){
	double normi=0.0;
	double normj=0.0;
	double dot_prod = 0.0;
	for (int ii=0;ii<L;ii++){
		dot_prod += loading_mat[ii][i] * loading_mat[ii][j];
		normi += pow(loading_mat[ii][i],2);
		normj += pow(loading_mat[ii][j],2);
	}

	return dot_prod/(sqrt(normi * normj));
}

void Loading_matrix(double loading_mat[][F]){
	// define Congruence matrix

	double A[F][F];
	init_congruence_mat(A, congruence);

	// column-orthogonal matrix
	double U[L][F]; 
	double Ut[F][L];
	initU(U);

	matmul(L, F, U, F, F, A, loading_mat);
	random_scaling_of_columns(loading_mat);

	printf("Colinearity: %f\n", get_colinearity(0,1, loading_mat));
	// for(int i=0; i<F; i++){
	// 	for(int j=0; j<F; j++){
	// 		printf("%f  ", get_colinearity(i, j, loading_mat));;
	// 	}
	// 	printf("\n");
	// }

}

int main(){
	srand(time(0));
	double A[L][F], B[L][F], C[L][F];
	Loading_matrix(A);
	Loading_matrix(B);
	Loading_matrix(C);
	
	// printf("Matrix A \n");
	// print_mat(L, F, A);
	// printf("Matrix B \n");
	// print_mat(L, F, B);
	// printf("Matrix C \n");
	// print_mat(L, F, C);

	// Multiply A, B, C to get the final tensor
	double tensor[L][L][L];
	double sum = 0.0;
	for(int i = 0; i<L; i++){
		for(int j=0; j<L; j++){
			for(int k=0; k<L; k++){
				tensor[i][j][k] = 0.0;
				for (int iter=0; iter<F; iter++){
					tensor[i][j][k] += A[i][iter]*B[j][iter]*C[k][iter];
					sum += pow(A[i][iter]*B[j][iter]*C[k][iter],2);
				}
			}
		}
	}
	sum = sqrt(sum);

	double noise_tensor[L][L][L];
	double eta = 0.05;
	double n_norm = randtensor_normal(L, L, L, noise_tensor);
	// Add noise
	for(int i = 0; i<L; i++){
		for(int j=0; j<L; j++){
			for(int k=0; k<L; k++){
				tensor[i][j][k] += eta*sum*noise_tensor[i][j][k]/n_norm;
			}
		}
	}

	//write in csv file
	char fileA[] = "matrix_A.csv"; char fileB[] = "matrix_B.csv"; char fileC[] = "matrix_C.csv";
	char filetensor[] = "Tensor.csv";
	FILE* fpA=fopen(fileA,"w+"); FILE* fpB=fopen(fileB,"w+"); FILE* fpC=fopen(fileC,"w+");
	FILE* fptensor=fopen(filetensor,"w+");

	for(int i=0; i<L; i++){
		for(int j=0; j<L; j++){
			for(int k=0; k<L; k++){
				if (k==L-1){
					fprintf(fptensor, "%f\n", tensor[i][j][k]);
				}else{
					fprintf(fptensor, "%f, ", tensor[i][j][k]);
				}
			}
		}fprintf(fptensor, "\n");
	}
	fclose(fptensor);

	for(int i=0; i<L; i++){
		for(int j=0; j<F; j++){
			if (j==F-1){
				fprintf(fpA, "%f\n", A[i][j]);
			}else{
				fprintf(fpA, "%f, ", A[i][j]);
			}
		}
	}
	fclose(fpA);

	for(int i=0; i<L; i++){
		for(int j=0; j<F; j++){
			if (j==F-1){
				fprintf(fpB, "%f\n", B[i][j]);
			}else{
				fprintf(fpB, "%f, ", B[i][j]);
			}
		}
	}
	fclose(fpB);

	for(int i=0; i<L; i++){
		for(int j=0; j<F; j++){
			if (j==F-1){
				fprintf(fpC, "%f\n", C[i][j]);
			}else{
				fprintf(fpC, "%f, ", C[i][j]);
			}
		}
	}
	fclose(fpC);



	// print the matrix
	// printf("\n Tensor: \n");
	// for(int i=0; i<L; i++)
	// 	print_mat(L, L, tensor[i]);

	return 0;
}