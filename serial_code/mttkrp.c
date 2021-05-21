#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#ifdef _OPENMP
#include<omp.h>
#endif

int n1,n2,n3,cp;


struct matrices
{
	double hat_A[100][100],hat_B[100][100],hat_C[100][100];	
};

struct matrices mttkrp(double X[n1][n2][n3], double A[n1][cp], double B[n2][cp], double C[n3][cp], int n1, int n2, int n3,int cp){
	/*
	Matricized tensor times Khatri-Rao product
	*/
	int i,j,k,f;

	struct matrices collection;

	//Ahat
	for(i=0;i<n1;i++){
		for(f=0;f<cp;f++){
			for(j=0;j<n2;j++){
				for(k=0;k<n3;k++){
					collection.hat_A[i][f] += X[i][j][k]*B[j][f]*C[k][f];
				}
			}
		}
	}

	//Bhat
	for(j=0;j<n2;j++){
		for(f=0;f<cp;f++){
			for(i=0;i<n1;i++){
				for(k=0;k<n3;k++){
					collection.hat_B[j][f] += X[i][j][k]*A[i][f]*C[k][f];
				}
			}
		}
	}

	//Chat
	for(k=0;k<n3;k++){
		for(f=0;f<cp;f++){
			for(i=0;i<n1;i++){
				for(j=0;j<n2;j++){
					collection.hat_C[k][f] += X[i][j][k]*A[i][f]*B[j][f];
				}
			}
		}
	}

	return collection;
}

void sample_init_X(double X[100][100][100], int n1, int n2, int n3){
	printf("\nEnter the values of Tensor X\n");

	for (k = 0; k < n3; ++k)
		{	printf("Slice %d\n",k);
			for (i = 0; i < n1; ++i)
			{
				for (j = 0; j < n2; ++j)
				{
					scanf("%lf",&X[i][j][k]);
					printf(" ");
				}
				printf("\n");
			}
			printf("\n\n");
		}
}

void sample_init_ABC(double A[100][100], double B[100][100], double C[100][100], int n1,int n2, int n3){
	printf("Enter the matr A\n");
	for (i = 0; i < n1; ++i)
		{
			for(f=0;f<cp;f++)
			{
				scanf("%lf",&A[i][f]);
				printf(" ");	
			}
			printf("\n");
		}	

	printf("Enter the matr B\n");
	for (j = 0; j < n2; ++j)
		{
			for(f=0;f<cp;f++)
			{
				scanf("%lf",&B[j][f]);
				printf(" ");
			}
			printf("\n");
		}

	printf("Enter the matr C\n");
	for (k = 0; k < n3; ++k)
		{
			for(f=0;f<cp;f++)
			{
				scanf("%lf",&C[k][f]);
				printf(" ");
			}
			printf("\n");
		}

}

void print_mat(double mat[][100], int s1, int s2){
	for(i=0;i<s1;i++){
		for(f=0;f<s2;f++){
			printf("%f ",mat[i][f]);
		}
		printf("\n");
	}
}

int main(int argc,char* argv[]){

	int i,j,k,f;

	printf("Enter the dimensions of the matrix: ");
	scanf("%d,%d,%d",&n1,&n2,&n3);
	printf("Enter number of components: ");
	scanf("%d",&cp);

	double X[n1][n2][n3],A[n1][cp],B[n2][cp],C[n3][cp];

	sample_init_X(X,n1,n2,n3);
	sample_init_ABC(A,B,C,n1,n2,n3);
	

	struct matrices collection = mttkrp(X,A,B,C,n1,n2,n3,cp);

	printf("hat_A\n");
	print_mat(collection.hat_A,n1,f);
	printf("hat_B\n");
	print_mat(collection.hat_B,n2,f);
	printf("hat_C\n");
	print_mat(collection.hat_C,n3,f);


	return 0;
}
