#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#ifdef _OPENMP
#include<omp.h>
#endif

#define eps 1e-6
// double congruence = 0.17;
static int gangs;

void mttkrp(int n1, int n2, int n3,int cp, int flag,double X[n1][n2][n3], double A[][cp], double B[][cp], double C[][cp], double hat_[][cp]){
	/*
	Matricized tensor times Khatri-Rao product
	*/

	// int i,j,k,f;
	// struct matrices collection;

	if(flag==0)
		//Ahat
		#pragma acc parallel loop present(X[:n1][:n2][:n3],hat_[:n1][:cp]) num_gangs(gangs) 
		for(int i=0;i<n1;i++){
			for(int f=0;f<cp;f++){
				double tmp = 0;
				// hat_[i][f] = 0;
				for(int j=0;j<n2;j++){
					for(int k=0;k<n3;k++){
						tmp += X[i][j][k]*B[j][f]*C[k][f];
						// hat_[i][f] += X[i][j][k]*B[j][f]*C[k][f];
					}
				}
				hat_[i][f] = tmp;
			}
		}


	if(flag==1)
	//Bhat
		#pragma acc parallel loop present(X[:n1][:n2][:n3],hat_[:n2][:cp]) num_gangs(gangs) 
		for(int j=0;j<n2;j++){
			for(int f=0;f<cp;f++){
				double tmp = 0;
				// hat_[j][f] = 0;
				for(int i=0;i<n1;i++){
					for(int k=0;k<n3;k++){
						tmp += X[i][j][k]*A[i][f]*C[k][f];
						// hat_[j][f] += X[i][j][k]*A[i][f]*C[k][f];
					}
				}
				hat_[j][f] = tmp;
			}
		}
	
	if(flag==2)
	//Chat
		#pragma acc parallel loop present(X[:n1][:n2][:n3],hat_[:n3][:cp]) num_gangs(gangs) 
		for(int k=0;k<n3;k++){
			for(int f=0;f<cp;f++){
				double tmp = 0;
				hat_[k][f] = 0;		
				for(int i=0;i<n1;i++){
					for(int j=0;j<n2;j++){
						tmp += X[i][j][k]*A[i][f]*B[j][f];
						// hat_[k][f] += X[i][j][k]*A[i][f]*B[j][f];
					}
				}
				hat_[k][f] = tmp;
			}
		}
}

void sample_init_X( int n1, int n2, int n3, double X[n1][n2][n3]){
	printf("\nEnter the values of Tensor X\n");

	for (int k = 0; k < n3; ++k)
		{	printf("Slice %d\n",k);
			for (int i = 0; i < n1; ++i)
			{
				for (int j = 0; j < n2; ++j)
				{
					scanf("%lf",&X[i][j][k]);
					printf(" ");
				}
				printf("\n");
			}
			printf("\n\n");
		}
}

void sample_init_ABC(int n1,int n2, int n3,int cp,double A[n1][cp], double B[n2][cp], double C[n3][cp]){	

	printf("Enter the matr B\n");
	for (int j = 0; j < n2; ++j)
		{
			for(int f=0;f<cp;f++)
			{
				scanf("%lf",&B[j][f]);
				printf(" ");
			}
			printf("\n");
		}

	printf("Enter the matr C\n");
	for (int k = 0; k < n3; ++k)
		{
			for(int f=0;f<cp;f++)
			{
				scanf("%lf",&C[k][f]);
				printf(" ");
			}
			printf("\n");
		}

}

void random_init_factors(int n1,int n2, int n3, int cp, int flag, double A[][cp], double B[][cp], double C[][cp]){
	// srand(time(0));
	if(flag==0){
		for(int j=0;j<n2;j++){
			for(int f = 0 ; f<cp ; f++){
				B[j][f] = rand() / (double) RAND_MAX;
			}
		}
		for(int k=0;k<n3;k++){
			for(int f = 0 ; f<cp ; f++){
				C[k][f] = rand() / (double) RAND_MAX;
			}
		}
	}
}

void print_mat(int s1, int s2,double mat[][s2]){
	for(int i=0;i<s1;i++){
		for(int f=0;f<s2;f++){
			printf("%f ",mat[i][f]);
		}
		printf("\n");
	}
}


double norm(double x[], int n){
	/*Matrix Frobenius Norm*/
	double sqsum = 0;
	//parallelizable with a reduction clause
	// #pragma acc parallel loop present(x[:n]) num_gangs(gangs) reduction(+:sqsum)
	for(int i =0; i<n; i++){
		sqsum += x[i]*x[i];
	}
	return sqrt(sqsum);
}

double tensor_norm(int n1,int n2, int n3, double X[][n2][n3]){
	double sqsum = 0;
	//parallelizable with a reduction clause
	#pragma acc parallel loop present(X[:n1][:n2][:n3]) num_gangs(gangs) reduction(+:sqsum) 
	for(int i =0; i<n1; i++){
		for(int j = 0; j < n2; j++){
			for(int k = 0; k < n3; k++){
				sqsum += X[i][j][k]*X[i][j][k];
			}
		}
	}
	return sqrt(sqsum);
}

void transpose( int nrowA, int ncolA, double A[][ncolA], double A_t[][nrowA]){
	// Completely parallelizable
	#pragma acc parallel loop present(A[:nrowA][:ncolA],A_t[:ncolA][:nrowA]) num_gangs(gangs) 
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
	int i,j,k;
	// double sum;

	// Parallelizable 
	// #pragma acc parallel loop present(A[:nrowA][:ncolA],B[:nrowB][:ncolB],C[:nrowA][:ncolB]) num_gangs(gangs) 
	for (i=0; i<nrowA; i++){
		for (j=0; j<ncolB; j++){
			C[i][j] = 0.0;
			// require reduction
			for (k=0; k<nrowB; k++){
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
}


void vect_mat_mul(int s1, int s2,int sv,double vec[],double mat[][s2], double result[]){
	if(s1!=sv){
		printf("Not Compatible\n");
		exit(0);
	}
	int i,j;

	#pragma acc parallel loop present(mat[:s1][:s2],result[:s2]) num_gangs(gangs) 
	for(int j=0;j<s2;j++){
		double sum = 0;
		for(int i=0;i<sv;i++){
			sum += mat[i][j]*vec[i];
		}
		result[j] = sum;
	}
}

double dot_product(double vec1[],int sv1,double vec2[],int sv2){
	/*
		Calculates the dot product of two vectors
		Arguments:
		vec1[] : First vector
		sv1: size of first vector
		vec2[] : Second vector
		sv2: size of Second vector
		Returns 
		result - resulting dot product
	*/

	if(sv1!=sv2){
		printf("Not Compatible\n");
		exit(0);
	}
	int i;
	double result = 0;

	#pragma acc parallel loop num_gangs(gangs)
	for(int i=0;i<sv1;i++){
		result += vec1[i]*vec2[i];
	}

	return result;
}

// void copy(int nrowA, int ncolA, double A[][ncolA], double B[][ncolA]){
// 	/*Copy values of A to B. B should be of same size as A*/
// 	int i,j;

// 	//Completely parallelizable
// 	#pragma acc parallel loop present() num_gangs(gangs) 
// 	for (i=0;i<nrowA; i++)
// 		for(j=0; j<ncolA; j++)
// 			B[i][j] = A[i][j];
// }


void Hadamard(int nrow, int ncol, double A[][ncol], double B[][ncol], int symm_flag, double C[][ncol]){
	/* nrow - number of rows in A and B
	 ncol - number of columns in A and B
	 A is a two dimensional matrix
	 B - second two dimensional matrix
	 C - resultant matrix*/

	// Completely parallelizable
	if(symm_flag==0)
		#pragma acc parallel loop present(A[:nrow][:ncol],B[:nrow][:ncol],C[:nrow][:ncol]) num_gangs(gangs) 
		for(int i=0; i<nrow; i++)
			for(int j=0; j<ncol; j++)
				C[i][j] = A[i][j] * B[i][j];
	else
		#pragma acc parallel loop present(A[:nrow][:ncol],B[:nrow][:ncol],C[:nrow][:ncol]) num_gangs(gangs) 
		for(int i=0; i<nrow; i++)
			for(int j=0; j<=i; j++){
				C[i][j] = A[i][j] * B[i][j];
				C[j][i] = C[i][j];
			}
}
    
void Cholesky_Decomposition(int nrows, int ncols, double A[][ncols], double L[][ncols])
{
	if (nrows!=ncols){
		// Raise error 
		printf("Input should be a square matrix!");
	}
	int i,j,k;

	// initalize L
	// Loops are parallellizable
	#pragma acc parallel loop present(A[:nrows][:ncols],L[:nrows][:ncols]) num_gangs(gangs) 
	for(int i =0; i< nrows; i++)
		for(int j = 0; j<ncols; j++)
			L[i][j] = 0;		
 
    // Decomposing a matrix into Lower Triangular
    //Loop carried dependency and therefore not parallelizable along one of the loop
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;

            if (j == i) // summation for diagnols
            {
                for (int k = 0; k < j; k++)
                    sum += L[j][k]*L[j][k];
                L[j][j] = sqrt(fabs(A[j][j] - sum));
            } else {
 
                // Evaluating L(i, j) using L(j, j)
                for (int k = 0; k < j; k++)
                    sum += (L[i][k] * L[j][k]);
          		if(L[j][j]<eps)
          			L[j][j] = eps;
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
}

void inverse_lower_triangular(int nrowL, int ncolL, double L[][ncolL], double Linverse[][ncolL]){
	if (nrowL!=ncolL){
		// Raise error 
		printf("Input should be a square matrix!");
	}
	//First two for loops are parallelizable
	for (int i=0; i<nrowL; i++){
		for (int j=0; j<=i;j++){
			if (i==j){
				if(L[i][i]<eps)
          			L[i][i] = eps;
				Linverse[i][j] = 1/L[i][i];
			}else{
				double sum = 0; 
				// #pragma omp parallel for num_threads(thread_count) private(i,j,k) shared(L, Linverse) reduction(+:sum)
				for(int k =j; k<i;k++){
					sum += L[i][k]*Linverse[k][j];
				}
				if(L[i][i]<eps)
          			L[i][i] = eps;
				Linverse[i][j] = -sum/(L[i][i]);
				Linverse[j][i] = 0;
			}
		}
	}
}

void inverse_symmetric_matrix(int nrow, int ncol, double A[][ncol], double inv_A[][ncol]){
	//Check if the matrix is square
	double L[nrow][ncol];
	// Find Inverse of lower triangular matrix
	double Linverse[nrow][ncol];
	// Find transpose of the inverse of the lower triangular matrix
	double L_transpose_inverse[nrow][ncol];
	#pragma acc data present(A[:nrow][:ncol],inv_A[:nrow][:ncol]) create(L[:nrow][:ncol],Linverse[:nrow][:ncol],L_transpose_inverse[:nrow][:ncol]) 
	{
		Cholesky_Decomposition(nrow, ncol, A, L);

		
		inverse_lower_triangular(nrow, ncol, L, Linverse);

		
		transpose(nrow, ncol, Linverse, L_transpose_inverse);

		// Multiply to obtain the resulting inverse matrix
		matmul(nrow, ncol, L_transpose_inverse, ncol, nrow, Linverse, inv_A);
	}


}

double Z_norm(int n1, int n2, int n3, int cp,double AtA[][cp], double BtB[][cp], double C[n3][cp], double lambda[]){
	/*
		Calculates the norm of the reconstructed Matrix Z using the parts that make it
		Arguments:
		n1, n2, n3: dimensions of the tensor
		cp: number of components
		A, B, C: Factor Matrices
		lambda: Normalizing factor
		Returns: 
		fnorm_Z: Resulting norm of reconstructed matrix
	*/

	double fnorm_Z;

	//Initialize Arrays
	double CtC[cp][cp], tmp[cp][cp],full[cp][cp];
	double C_t[cp][n3],tmp2[cp];
	#pragma acc data present(AtA[:cp][:cp],BtB[:cp][:cp],C[:n3][:cp]) create(CtC[:cp][:cp], tmp[:cp][:cp],full[:cp][:cp],C_t[:cp][:n3],tmp2[:cp],fnorm_Z) 
	{
		//Transpose Matrix C
		transpose(n3,cp,C,C_t);

		//Calculate Ctc
		matmul(cp, n3, C_t, n3, cp, C, CtC);

		//Hadamard product of all three
		Hadamard(cp,cp,CtC,BtB,1,tmp);
		Hadamard(cp,cp,tmp,AtA,1,full);

		//lam.T hadamard lam
		vect_mat_mul(cp,cp,cp,lambda,full,tmp2);
		fnorm_Z = dot_product(lambda,cp,tmp2,cp);
	}

	return sqrt(fnorm_Z);
}

double XZ_product(int n1, int n2, int n3, int cp, double X[n1][n2][n3],double A[n1][cp],double B[n2][cp],double C[n3][cp], double lambda[]){
	/*
		Calculates the dot product of the original and reconstructed Matrix 
		Arguments:
		n1, n2, n3: dimensions of the tensor
		cp: number of components
		X: Original Tensor
		A, B, C: Factor Matrices
		lambda: Normalizing factor
		Returns: 
		fnorm_XZ: Resulting dot product
	*/
	double fnorm_XZ = 0;

	#pragma acc parallel loop present(X[:n1][:n2][:n3],A[:n1][:cp],B[:n2][:cp],C[:n3][cp]) num_gangs(gangs) reduction(+:fnorm_XZ)
	for(int f = 0; f<cp;f++){
		for(int i=0;i<n1;i++){
			for(int j=0;j<n2;j++){
				for(int k=0;k<n3;k++){
					fnorm_XZ += X[i][j][k]*A[i][f]*B[j][f]*C[k][f]*lambda[f];
				}
			}
		}
	}

	return fnorm_XZ;
}

void reconstruct(int n1, int n2, int n3, int cp, double A[][cp], double B[][cp], double C[][cp],double lambda[],double Z[n1][n2][n3]){
	/*
		Reconstruct tensors using its parts
		Arguments
		n1,n2,n3: Tensor dimensions
		A,B,C: Factor matrices
		lambda: Normalizing factors
		Z: Reconstructed tensor pointer which gets updated in this function
		Returns:
		None
	*/
	int i,j,k,f;

	// #pragma acc parallel loop present() num_gangs(gangs) 
	for(int i=0;i<n1;i++){
		for(int j=0;j<n2;j++){
			for(int k=0;k<n3;k++){
				Z[i][j][k] = 0;
				for(int f=0;f<cp;f++){
					Z[i][j][k] += lambda[f]*A[i][f]*B[j][f]*C[k][f];
				}
			}
		}
	}

}


void col_2_norm(int s1, int s2,double mat[][s2],double lambda[]){
	/*
		Normalizes the columns of a matrix using 2-norm and stores it in lambda
		Arguments:
		s1,s2: matrix sizes
		mat: matrix
		lambda: pointer which will get updated
		Returns: None
	*/

	#pragma acc parallel loop present(mat[:s1][:s2]) num_gangs(gangs) 
	for(int j=0;j<s2;j++){
		lambda[j] = 0;
		for(int i=0;i<s1;i++){
			lambda[j] += mat[i][j]*mat[i][j];
		}
		lambda[j] = sqrt(lambda[j]);
		for(int i=0;i<s1;i++){
			mat[i][j] /= lambda[j];
		}
	}
}

void print_tensor(int n1, int n2, int n3, double tens[][n2][n3]){
	for(int k=0;k<n3;k++){
		printf("Slice %d\n",k);
		for (int i = 0; i < n1; ++i)
		{
			for (int j = 0; j < n2; ++j)
			{
				printf("%f ", tens[i][j][k]);
			}
			printf("\n");
		}
	}
}

void max_col_norm(int s1, int s2, double mat[][s2], double lambda[]){
	/*
		Normalizes the columns of a matrix using max-norm and stores it in lambda
		Arguments:
		s1,s2: matrix sizes
		mat: matrix
		lambda: pointer which will get updated
		Returns: None
	*/
	double max_val;
	#pragma acc parallel loop present(mat[:s1][:s2]) num_gangs(gangs) 
	for(int j=0;j<s2;j++){
		lambda[j] = 0;
		max_val = 1;
		for(int i=0;i<s1;i++){
			if(fabs(mat[i][j])>max_val)
				max_val = fabs(mat[i][j]);
		}
		lambda[j] = max_val;
		for(int i=0;i<s1;i++){
			mat[i][j] /= lambda[j];
		}
	}

}

// void load_tensor_from_csv(int n1, int n2, int n3, double X[][n2][n3]){
// 	FILE *fstream = fopen("data50/Tensor.csv","r");	
// 	char *record,*line;
// 	if(fstream == NULL){
//       	printf("\n File Not Found");
//       	return -1 ;
//    	}
//    	int tmp;
//   //  	while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL){
// 		// record = strtok(line,",");
// 		// while(record != NULL)
// 		// {
// 		// 	mat[i][j++] = atoi(record) ;
// 		// 	record = strtok(NULL,",");
// 		// }
// 		// ++i ;
//   //  }

//    	for(int i=0;i<n1;i++){
//    		for(int j=0;j<n2;j++){
//    			for(int k=0;k<n3;k++){
//    				fscanf(fstream,"%i",tmp)
//    				printf("%s\n", tmp);
//    			}
//    			printf("\n");
//    		}
// 		printf("\n");
//    	}
// }

void gramschmith(int i, int L, int F, double U[][F]){
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
void random_scaling_of_columns(int L, int F, double U[][F]){
	int i,k;
	double randnum;
	// #pragma omp parallel for num_threads(thread_count) default(none) shared(L,F,U) private(randnum, i, k)
	for(i=0; i<F; i++){
		randnum = 1 +(double)(rand() % 9); 
		for (int k=0; k<L; k++){
			U[k][i] = randnum * U[k][i];
		}
	}
}

double get_colinearity(int i, int j, int L, int F, double loading_mat[][F]){
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

void randmat(int nrows, int ncols, double mat[][ncols]){
	// Non-parallelizable
	for(int i=0; i<nrows; i++){
		for(int j=0; j<ncols; j++){
			mat[i][j] = (double)(rand() % 10000) / 10000.0;
	}}
}

void initU(int L, int F, double U[][F]){
	randmat(L, F, U);
	// Gram-schmidth orthogonalization
	for(int i=0; i<F; i++){
		gramschmith(i, L, F, U);

		//Normalization
		double norm = 0;
		for (int k =0; k<L; k++){
			norm = norm + pow(U[k][i],2);
		}norm = sqrt(norm);

		while(norm<0.001){
			// printf("Need Reinitialization");
			for(int k=0;k<L;k++){
				U[k][i] = (double)(rand() % 10000) / 10000.0;
			}
			gramschmith(i, L,F,U);
			double norm = 0;
			for (int k =0; k<L; k++){
				norm += pow(U[k][i],2);
			}norm =sqrt(norm);
		}

		// parallelizable
		int k;
		for (k =0; k<L; k++){
			U[k][i] = U[k][i]/norm;
		}
	}
	
}

void init_congruence_mat(int F,double A[][F], double congruence){

	//parallelizable loops
	// #pragma acc parallel loop present(A[:F][:F]) num_gangs(gangs) 
	for (int i=0; i<F;i++){
		for (int j = 0; j<F; j++){
			if (i==j){
				A[i][j] = 1.0;
			}else{
				A[i][j] = congruence;
			}
		}
	}
}

void Loading_matrix(int L, int F, double congruence, double loading_mat[][F]){
	// define Congruence matrix

	double A[F][F];
	init_congruence_mat(F, A, congruence);

	// column-orthogonal matrix
	double U[L][F]; 
	initU(L,F,U);

	matmul(L, F, U, F, F, A, loading_mat);
	random_scaling_of_columns(L,F,loading_mat);

	// printf("Colinearity: %f\n", get_colinearity(0,1,L,F, loading_mat));

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

void generate_data(int n1, int n2, int n3, int cp, double congruence, double tensor[][n2][n3]){
	// srand(time(0));
	int L = n1;
	int F = cp;
	double A[L][F], B[L][F], C[L][F];
	Loading_matrix(L,F, congruence,A);
	Loading_matrix(L,F, congruence,B);
	Loading_matrix(L,F,congruence,C);

	// Multiply A, B, C to get the final tensor
	// double tensor[L][L][L];
	int i,j,k, iter;

	double sum = 0.0;
	// #pragma acc parallel loop present() num_gangs(gangs) 
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
	// #pragma acc parallel loop present() num_gangs(gangs) 
	for(int i = 0; i<L; i++){
		for(int j=0; j<L; j++){
			for(int k=0; k<L; k++){
				tensor[i][j][k] += eta*sum*noise_tensor[i][j][k]/n_norm;
			}
		}
	}

}



int main(int argc,char* argv[]){

	/*
		Arguments: 
		n1,n2,n3: Size
		F: Rank for synthesis
		congruence: For synthesis
		cp: Rank for reconstructing
		max_iter: Maximum Number of Iterations
		tol: Error Tolerance
	*/

	int cp,n1,n2,n3,F;
	long int max_iter = 100000, ite = 0;
	double tol = 1e-6, congruence;

	if(argc<=6 && argc > 9){
		printf("Incorrect Number of Arguments\n");
		exit(0);
	}
	else{
		n1 = atoi(argv[1]);
		n2 = atoi(argv[2]);
		n3 = atoi(argv[3]);
		F = atoi(argv[4]);
		congruence = atof(argv[5]);
		cp = atoi(argv[6]);
		max_iter = atoi(argv[7]);
		tol = atof(argv[8]);
		gangs = atoi(argv[9]);
	}


	double X[n1][n2][n3],A[n1][cp],B[n2][cp],C[n3][cp];
	double A_t[cp][n1],B_t[cp][n2],C_t[cp][n3];
	double Ata[cp][cp], Btb[cp][cp], Ctc[cp][cp];
	double had_1[cp][cp], had_2[cp][cp], had_3[cp][cp];
	double inv_had_1[cp][cp], inv_had_2[cp][cp], inv_had_3[cp][cp];
	double Z[n1][n2][n3], lambda[cp];
	double hat_A[n1][cp], hat_B[n2][cp], hat_C[n3][cp];
	double error,fit;

	srand(1710);

	// double	t1syn = omp_get_wtime();
    // #pragma acc data create(X) copyin(n1,n2,n3,F) 
	generate_data(n1,n2,n3,F,congruence,X);
	// double	t2syn = omp_get_wtime();


	random_init_factors(n1,n2,n3,cp,0,A,B,C);
    #pragma acc data copyin(n1,n2,n3,cp,ite,max_iter,tol,X[:n1][:n2][:n3],A[:n1][:cp],B[:n2][:cp],C[:n3][:cp]) copyout(ite,error,fit) \
    	create(A_t[:cp][:n1],B_t[:cp][:n2],C_t[:cp][:n3], Ata[:cp][:cp], Btb[:cp][:cp], Ctc[:cp][:cp],had_1[:cp][:cp], had_2[:cp][:cp], had_3[:cp][:cp], \
   			inv_had_1[:cp][:cp], inv_had_2[:cp][:cp], inv_had_3[:cp][:cp],hat_A[:n1][:cp], hat_B[:n2][:cp], hat_C[:n3][:cp],lambda[:cp]) 
    {
	double X_norm = tensor_norm(n1,n2,n3,X), Xhat_norm, prod_XZ;
	double fit_old = 5;
	fit = 1, error = 1;
	while(ite<max_iter && fabs(fit-fit_old)>tol){
		ite += 1;
		//Computing A
		//MTTKRP with A
		mttkrp(n1,n2,n3,cp,0,X,A,B,C,hat_A);
		//Calculation of (BtB*CtC)^(-1)
		transpose(n2,cp,B,B_t);
		transpose(n3,cp,C,C_t);
		matmul(cp, n2, B_t, n2, cp, B, Btb);
		matmul(cp, n3, C_t, n3, cp, C, Ctc);
		Hadamard(cp,cp,Ctc,Btb,1,had_1);
		inverse_symmetric_matrix(cp,cp,had_1,inv_had_1);
		
		//Multiplying both terms
		matmul(n1,cp,hat_A,cp,cp,inv_had_1,A);
		
		//Normalizing and updating lambda
		if(ite==1)
			col_2_norm(n1,cp,A,lambda);
		else
			max_col_norm(n1,cp,A,lambda);

		//Computing B
		//MTTKRP with B
		mttkrp(n1,n2,n3,cp,1,X,A,B,C,hat_B);
		//Calculation of (AtA*CtC)^(-1)
		transpose(n1,cp,A,A_t);
		transpose(n3,cp,C,C_t);
		matmul(cp, n1, A_t, n1, cp, A, Ata);
		matmul(cp, n3, C_t, n3, cp, C, Ctc);
		Hadamard(cp,cp,Ctc,Ata,1,had_2);
		inverse_symmetric_matrix(cp,cp,had_2,inv_had_2);
		//Multiplying both terms
		matmul(n2,cp,hat_B,cp,cp,inv_had_2,B);
		//Normalizing and updating lambda
		if(ite==1)
			col_2_norm(n2,cp,B,lambda);
		else
			max_col_norm(n2,cp,B,lambda);

		//Computing C
		//MTTKRP with C
		mttkrp(n1,n2,n3,cp,2,X,A,B,C,hat_C);
		//Calculation of (AtA*BtB)^(-1)
		transpose(n1,cp,A,A_t);
		transpose(n2,cp,B,B_t);
		matmul(cp, n1, A_t, n1, cp, A, Ata);
		matmul(cp, n2, B_t, n2, cp, B, Btb);
		Hadamard(cp,cp,Btb,Ata,1,had_3);
		inverse_symmetric_matrix(cp,cp,had_3,inv_had_3);
		//Multiplying both terms
		matmul(n3,cp,hat_C,cp,cp,inv_had_3,C);
		//Normalizing and updating lambda
		double lambda[cp];
		if(ite==1)
			col_2_norm(n3,cp,C,lambda);
		else
			max_col_norm(n3,cp,C,lambda);

		//Error Calculation
		Xhat_norm = Z_norm(n1,n2,n3,cp,Ata,Btb,C,lambda);
		prod_XZ = XZ_product(n1,n2,n3,cp,X,A,B,C,lambda);
		
		error = sqrt(fabs(X_norm*X_norm + Xhat_norm*Xhat_norm - 2*prod_XZ));

		fit_old = fit;
		fit = 1 - error/X_norm;
		if(ite%100==0)
			printf("Iteration: %ld, Error: %f, Fit: %f\n",ite, error, fit);
	}
}
	// // Uncomment below lines to print the factor matrices, weights and reconstructed Tensor
	// printf("Final\n");
	// printf("A\n");
	// print_mat(n1,cp,A);
	// printf("\n\n");
	// printf("B\n");
	// print_mat(n2,cp,B);
	// printf("\n\n");
	// printf("C\n");
	// print_mat(n3,cp,C);	
	// printf("\n\n");	
	// for (int i = 0; i < cp; ++i)
	// {
	// 	printf("%f\n", lambda[i]);
	// }
	// reconstruct(n1,n2,n3,cp,A,B,C,lambda,Z);
	// printf("\nReconstructed Tensor\n\n");
	// print_tensor(n1,n2,n3,Z);
	printf("\n");
	printf("Final Error: %f\n", error);
	printf("Final Fit: %f\n", fit);
	printf("Number of Iterations: %ld\n", ite);

	return 0;
}