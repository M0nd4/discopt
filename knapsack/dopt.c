#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[])
{
	FILE *fi;
	int k;
	int i, n;
	int obj[100][2];  // max = 100 objects
	char taken[100];
	int weight;
	int value;

	if(argc==2){
		// check filename:
		// printf("filename = %s\n", argv[1]);

		// read data file:
		fi = fopen(argv[1], "rt");
		fscanf(fi, "%d %d\n", &n, &k);
		for(i=0;i<n;i++)
			fscanf(fi, "%d %d\n", &obj[i][0], &obj[i][1]);
		fclose(fi);

		// solve problem:
		memset(taken,'0',sizeof(taken));
		value=weight=0;
		for(i=0;i<n;i++){
			if(obj[i][1]<=k-weight){ // if object fits, take it:
				taken[i] = '1';
				weight += obj[i][1];
				value += obj[i][0];
			}
		}

		// write answer to standard output:
		printf("%d %d\n", value, 0);
		printf("%c",taken[0]);
		for(i=1;i<n;i++)
			printf(" %c",taken[i]);
		printf("\n");
	}
	return 0;
}
