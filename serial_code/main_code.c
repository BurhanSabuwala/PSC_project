#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#ifdef _OPENMP
#include<omp.h>
#endif

// DEPRECATED!!!

// int n1,n2,n3,cp;

// struct matrices
// {	
// 	double hat_A[100][100],hat_B[100][100],hat_C[100][100];	
// };

void mttkrp(int n1, int n2, int n3,int cp, int flag,double X[n1][n2][n3], double A[n1][cp], double B[n2][cp], double C[n3][cp], double hat_[][cp]){
	/*
	Matricized tensor times Khatri-Rao product
	*/
	int i,j,k,f;

	// struct matrices collection;

	if(flag==0)
		//Ahat
		for(i=0;i<n1;i++){
			for(f=0;f<cp;f++){
				for(j=0;j<n2;j++){
					for(k=0;k<n3;k++){
						// collection.hat_A[i][f] += X[i][j][k]*B[j][f]*C[k][f];
						hat_[i][f] += X[i][j][k]*B[j][f]*C[k][f];
					}
				}
			}
		}


	if(flag==1)
	//Bhat
		for(j=0;j<n2;j++){
			for(f=0;f<cp;f++){
				for(i=0;i<n1;i++){
					for(k=0;k<n3;k++){
						hat_[j][f] += X[i][j][k]*A[i][f]*C[k][f];
					}
				}
			}
		}
	
	if(flag==2)
	//Chat
		for(k=0;k<n3;k++){
			for(f=0;f<cp;f++){
				for(i=0;i<n1;i++){
					for(j=0;j<n2;j++){
						hat_[k][f] += X[i][j][k]*A[i][f]*B[j][f];
					}
				}
			}
		}

	// return collection;
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
	printf("Enter the matr A\n");
	for (int i = 0; i < n1; ++i)
		{
			for(int f=0;f<cp;f++)
			{
				scanf("%lf",&A[i][f]);
				printf(" ");	
			}
			printf("\n");
		}	

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

double tensor_norm(double X[][100][100], int n1,int n2, int n3){
	double sqsum = 0;

	//parallelizable with a reduction clause
	for(int i =0; i<n1; i++){
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

void mat_vect_mul(double mat[][100],int s1, int s2, double vec[],int sv, double result[]){
	if(s2!=sv){
		printf("Not Compatible\n");
		exit(0);
	}

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

	for(int j=0;j<s2;j++){
		result[j] = 0;
		for(int i=0;i<sv;i++){
			result[j] += mat[i][j]*vec[i];
		}
	}
}

double dot_product(double vec1[],int sv1,double vec2[],int sv2){

	if(sv1!=sv2){
		printf("Not Compatible\n");
		exit(0);
	}

	double result = 0;
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

void inverse_lower_triangular(int nrowL, int ncolL, double L[][ncolL], double Linverse[][ncolL]){
	if (nrowL!=ncolL){
		// Raise error 
		printf("Input should be a square matrix!");
	}
	//First two for loops are parallelizable
	for (int i=0; i<nrowL; i++){
		for (int j=0; j<=i;j++){
			if (i==j){
				Linverse[i][j] = 1/L[i][i];
			}else{
				double sum = 0;
				for(int k =j; k<i;k++){
					sum += L[i][k]*Linverse[k][j];
				}
				Linverse[i][j] = -sum/(L[i][i]);
			}
		}
	}
}

void inverse_symmetric_matrix(int nrow, int ncol, double A[][ncol], double inv_A[][nrow]){
	//Check if the matrix is square
	double L[nrow][ncol];
	Cholesky_Decomposition(nrow, ncol, A, L, 0);

	// Find Inverse of lower triangular matrix
	double Linverse[nrow][ncol];
	inverse_lower_triangular(nrow, ncol, L, Linverse);

	// Find transpose of the inverse of the lower triangular matrix
	double L_transpose_inverse[nrow][ncol];
	transpose(nrow, ncol, Linverse, L_transpose_inverse);

	// Multiply to obtain the resulting inverse matrix
	matmul(nrow, ncol, L_transpose_inverse, ncol, nrow, Linverse, inv_A);

}

double Z_norm(int n1, int n2, int n3, int cp,double A[n1][cp], double B[n2][cp], double C[n3][cp], double lambda[]){

	double fnorm_Z;

	double Ata[cp][cp], Btb[cp][cp], Ctc[cp][cp], tmp[cp][cp],full[cp][cp];
	double A_t[cp][n1], B_t[cp][n2], C_t[cp][n3],tmp2[cp];

	transpose(n1,cp,A,A_t);
	transpose(n2,cp,B,B_t);
	transpose(n3,cp,C,C_t);

	matmul(cp, n1, A_t, n1, cp, A, Ata);
	matmul(cp, n2, B_t, n2, cp, B, Btb);
	matmul(cp, n3, C_t, n3, cp, C, Ctc);


	Hadamard(cp,cp,Ctc,Btb,tmp);
	Hadamard(cp,cp,tmp,Ata,full);

	vect_mat_mul(cp,cp,cp,lambda,full,tmp2);
	fnorm_Z = dot_product(lambda,cp,tmp2,cp);

	return sqrt(fnorm_Z);
}

double XZ_norm(int n1, int n2, int n3, int cp, double X[n1][n2][n3],double A[n1][cp],double B[n2][cp],double C[n3][cp], double lambda[]){
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


void compute_lambda(int s1, int s2,double mat[][s2],double lambda[]){
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
				/* code */
			}
			printf("\n");
		}
	}
}


int main(int argc,char* argv[]){

	int i,j,k,f,cp,n1,n2,n3, max_iter = 100, ite = 0;
	// struct matrices collection;

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
	double t_hatA[cp][n1], t_hatB[cp][n2], t_hatC[cp][n3];
	double hat_A[n1][cp], hat_B[n2][cp], hat_C[n3][cp];

	sample_init_X(n1,n2,n3,X);
	sample_init_ABC(n1,n2,n3,cp,A,B,C);
	
	double X_norm = tensor_norm(X,n1,n2,n3), Xhat_norm, norm_XZ, error;

	while(ite<max_iter){
		ite += 1;
		mttkrp(n1,n2,n3,cp,0,X,A,B,C,hat_A);
		mttkrp(n1,n2,n3,cp,1,X,A,B,C,hat_B);
		mttkrp(n1,n2,n3,cp,2,X,A,B,C,hat_C);

		// mttkrp(n1,n2,n3,cp,X,A,B,C,hat_A);

		// if(ite==1){
		// 	printf("Iteration %d\n",ite);
		// 	print_mat(n1,cp,hat_A);
		// 	printf("\n\n");
		// 	print_mat(n2,cp,hat_B);
		// 	printf("\n\n");
		// 	print_mat(n3,cp,hat_C);	
		// 	printf("\n\n");	
		// }

		transpose(n1,cp,hat_A,t_hatA);
		transpose(n2,cp,hat_B,t_hatB);
		transpose(n3,cp,hat_C,t_hatC);

		transpose(n1,cp,A,A_t);
		transpose(n2,cp,B,B_t);
		transpose(n3,cp,C,C_t);

		matmul(cp, n1, A_t, n1, cp, A, Ata);
		matmul(cp, n2, B_t, n2, cp, B, Btb);
		matmul(cp, n3, C_t, n3, cp, C, Ctc);

		Hadamard(cp,cp,Ctc,Btb,had_1);
		Hadamard(cp,cp,Ctc,Ata,had_2);
		Hadamard(cp,cp,Btb,Ata,had_3);

		inverse_symmetric_matrix(cp,cp,had_1,inv_had_1);
		inverse_symmetric_matrix(cp,cp,had_2,inv_had_2);
		inverse_symmetric_matrix(cp,cp,had_3,inv_had_3);

		matmul(cp,cp,inv_had_1,cp,n1,t_hatA,A_t);
		transpose(cp,n1,A_t,A);
		matmul(cp,cp,inv_had_2,cp,n2,t_hatB,B_t);
		transpose(cp,n2,B_t,B);
		matmul(cp,cp,inv_had_3,cp,n3,t_hatC,C_t);
		transpose(cp,n3,C_t,C);

		// if(ite==2){
		// 	print_mat(cp,n3,collection.hat_C);
		// 	printf("\n");
		// }

		compute_lambda(n1,cp,A,lambda);
		compute_lambda(n2,cp,B,lambda);
		double lambda[cp];
		compute_lambda(n3,cp,C,lambda);

		Xhat_norm = Z_norm(n1,n2,n3,cp,A,B,C,lambda); // Lambda?
		norm_XZ = XZ_norm(n1,n2,n3,cp,X,A,B,C,lambda);
		// printf("%f, %f, %f\n",X_norm,Xhat_norm,norm_XZ);

		error = sqrt(X_norm*X_norm + Xhat_norm*Xhat_norm - 2*norm_XZ);
		printf("Iteration: %d, Error: %f\n",ite, error);
		// break;

	}

	printf("Final\n");
	print_mat(n1,cp,A);
	printf("\n\n");
	print_mat(n2,cp,B);
	printf("\n\n");
	print_mat(n3,cp,C);	
	printf("\n\n");	
	for (int i = 0; i < cp; ++i)
	{
		printf("%f\n", lambda[i]);
	}
	// printf("Final Error: %f\n", error);
	reconstruct(n1,n2,n3,cp,A,B,C,lambda,Z);
	print_tensor(n1,n2,n3,Z);

	return 0;
}
