#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#ifdef _OPENMP
#include<omp.h>
#endif

#define eps 1e-6

void mttkrp(int n1, int n2, int n3,int cp, int flag,double X[n1][n2][n3], double A[][cp], double B[][cp], double C[][cp], double hat_[][cp]){
	/*
	Matricized tensor times Khatri-Rao product
	*/

	int i,j,k,f;

	// struct matrices collection;

	if(flag==0){
		//Ahat
		#pragma acc parallel loop
		for(i=0;i<n1;i++){
			#pragma acc loop
			for(f=0;f<cp;f++){
				hat_[i][f] = 0;
				for(j=0;j<n2;j++){
					for(k=0;k<n3;k++){
						hat_[i][f] += X[i][j][k]*B[j][f]*C[k][f];
					}
				}
			}
		}
	}


	if(flag==1){
		//Bhat
		#pragma acc parallel loop
		for(j=0;j<n2;j++){
			#pragma acc loop
			for(f=0;f<cp;f++){
				hat_[j][f] = 0;
				for(i=0;i<n1;i++){
					for(k=0;k<n3;k++){
						hat_[j][f] += X[i][j][k]*A[i][f]*C[k][f];
					}
				}
			}
		}
	}
	
	if(flag==2){
		//Chat
		#pragma acc parallel loop
		for(k=0;k<n3;k++){
			#pragma acc loop
			for(f=0;f<cp;f++){
				hat_[k][f] = 0;		
				for(i=0;i<n1;i++){
					for(j=0;j<n2;j++){
						hat_[k][f] += X[i][j][k]*A[i][f]*B[j][f];
					}
				}
			}
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
	srand(time(0));
	if(flag==0){
		#pragma acc parallel loop
		for(int j=0;j<n2;j++){
			#pragma acc loop
			for(int f = 0 ; f<cp ; f++){
				B[j][f] = rand() / (double) RAND_MAX;
			}
		}
		#pragma acc parallel loop
		for(int k=0;k<n3;k++){
			#pragma acc loop	
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
	#pragma acc parallel loop
	for(int i =0; i<n; i++){
		sqsum += x[i]*x[i];
	}
	return sqrt(sqsum);
}

double tensor_norm(int n1,int n2, int n3,double X[][n2][n3]){
	double sqsum = 0;
	
	//parallelizable with a reduction clause
	#pragma acc parallel loop
	for(int i =0; i<n1; i++){
		#pragma acc loop
		for(int j = 0; j < n2; j++){
			for(int k = 0; k < n3; k++){
				sqsum += X[i][j][k]*X[i][j][k];
			}
		}
	}
	return sqrt(sqsum);
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

void transpose( int nrowA, int ncolA, double A[][ncolA], double A_t[][nrowA]){

	// Completely parallelizable
	#pragma acc parallel loop
	for (int i=0; i<nrowA; i++)
		#pragma acc loop
		for (int j=0; j<ncolA; j++)
			A_t[j][i] = A[i][j];
}

void matmul(int nrowA, int ncolA, double A[][ncolA], int nrowB, int ncolB, double B[][ncolB], double C[][ncolB]){
	/*Matrix multiplication*/
	if (ncolA != nrowB){
		// raise error
		printf("Matrix dimensions do not match for multiplication! \n");
	}
			
	// Parallelizable 
	#pragma acc parallel loop
	for (int i=0; i<nrowA; i++){
		for (int j=0; j<ncolB; j++){
			C[i][j] = 0;
			// require reduction
			for (int k=0; k<nrowB; k++){
				C[i][j] = C[i][j] + A[i][k]*B[k][j];
			}
		}
	}
}

void mat_vect_mul(int s1, int s2, int sv,double mat[][s2], double vec[], double result[]){
	if(s2!=sv){
		printf("Not Compatible\n");
		exit(0);
	}
	#pragma acc parallel loop
	for(int i=0;i<s1;i++){
		result[i] = 0;
		for(int j=0;j<s2;j++){
			result[i] += mat[i][j];
		}
	}
}

void vect_mat_mul(int s1, int s2,int sv,double vec[],double mat[][s2], double result[]){
	if(s1!=sv){
		printf("Not Compatible\n");
		exit(0);
	}
	#pragma acc parallel loop
	for(int j=0;j<s2;j++){
		result[j] = 0;
		for(int i=0;i<sv;i++){
			result[j] += mat[i][j]*vec[i];
		}
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

	double result = 0;
	#pragma acc parallel loop
	for(int i=0;i<sv1;i++){
		result += vec1[i]*vec2[i];
	}

	return result;
}

void copy(int nrowA, int ncolA, double A[][ncolA], double B[][ncolA]){
	/*Copy values of A to B. B should be of same size as A*/

	//Completely parallelizable
	for (int i=0;i<nrowA; i++)
		for(int j=0; j<ncolA; j++)
			B[i][j] = A[i][j];
}


void Hadamard(int nrow, int ncol, double A[][ncol], double B[][ncol], int symm_flag, double C[][ncol]){
	/* nrow - number of rows in A and B
	 ncol - number of columns in A and B
	 A is a two dimensional matrix
	 B - second two dimensional matrix
	 C - resultant matrix*/

	// Completely parallelizable
	if(symm_flag==0){
		#pragma acc parallel loop
		for(int i=0; i<nrow; i++)
			#pragma acc loop
			for(int j=0; j<ncol; j++)
				C[i][j] = A[i][j] * B[i][j];
	}
	else
		#pragma acc parallel loop
		for(int i=0; i<nrow; i++)
			#pragma acc loop
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

	// initalize L
	// Loops are parallellizable

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
	Cholesky_Decomposition(nrow, ncol, A, L);

	// Find Inverse of lower triangular matrix
	double Linverse[nrow][ncol];
	inverse_lower_triangular(nrow, ncol, L, Linverse);

	// Find transpose of the inverse of the lower triangular matrix
	double L_transpose_inverse[nrow][ncol];
	transpose(nrow, ncol, Linverse, L_transpose_inverse);

	// Multiply to obtain the resulting inverse matrix
	matmul(nrow, ncol, L_transpose_inverse, ncol, nrow, Linverse, inv_A);


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

int main(int argc,char* argv[]){

	int cp,n1,n2,n3, max_iter = 1000, ite = 0;

	printf("Enter the dimensions of the tensor: ");
	scanf("%d,%d,%d",&n1,&n2,&n3);
	printf("Enter the rank: ");
	scanf("%d",&cp);

	double X[n1][n2][n3],A[n1][cp],B[n2][cp],C[n3][cp];
	double A_t[cp][n1],B_t[cp][n2],C_t[cp][n3];
	double Ata[cp][cp], Btb[cp][cp], Ctc[cp][cp];
	double had_1[cp][cp], had_2[cp][cp], had_3[cp][cp];
	double inv_had_1[cp][cp], inv_had_2[cp][cp], inv_had_3[cp][cp];
	double Z[n1][n2][n3], lambda[cp];
	double hat_A[n1][cp], hat_B[n2][cp], hat_C[n3][cp];

	sample_init_X(n1,n2,n3,X);
	random_init_factors(n1,n2,n3,cp,0,A,B,C);
	double X_norm = tensor_norm(n1,n2,n3,X), Xhat_norm, prod_XZ, error=1;

	double tol = 1e-6, fit;

	while(ite<max_iter && error>tol){
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

		fit = 1 - error/X_norm;

		printf("Iteration: %d, Error: %f, Fit: %f\n",ite, error, fit);
	}

	printf("Final\n");
	printf("A\n");
	print_mat(n1,cp,A);
	printf("\n\n");
	printf("B\n");
	print_mat(n2,cp,B);
	printf("\n\n");
	printf("C\n");
	print_mat(n3,cp,C);	
	printf("\n\n");	
	for (int i = 0; i < cp; ++i)
	{
		printf("%f\n", lambda[i]);
	}
	printf("\n");
	printf("Final Error: %f\n", error);
	printf("Final Fit: %f\n", fit);
	reconstruct(n1,n2,n3,cp,A,B,C,lambda,Z);
	printf("\nReconstructed Tensor\n\n");
	print_tensor(n1,n2,n3,Z);

	return 0;
}
