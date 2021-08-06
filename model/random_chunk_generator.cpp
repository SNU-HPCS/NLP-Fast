#include <mkl.h>
#include <sys/stat.h>
#include "c_matrix.hpp"

int main(int argc, char *argv[]) {
	int elem_num;
	int layer_num;
	float *chunk;
	FILE *fp;
	struct stat st = {};
	char fname[1024];

	if (argc != 4) {
		fprintf(stderr, "Usage: %s [chunk_directory] [size of random chunk] [number of layers]\n", argv[0]);
		return -1;
	}

	if (stat(argv[1], &st) == 0){
		if ((st.st_mode & S_IFDIR) == 0) {
			fprintf(stderr, "directory (%s) does not exist\n", argv[1]);
			return -1;
		}
	}

	if ((elem_num = atoi(argv[2])) == 0) {
		return -1;
	}
	if ((layer_num = atoi(argv[3])) == 0) {
		return -1;
	}

	chunk = mat_rand<float>(1, elem_num);

	for (int i = 0; i < layer_num; i++) {
		snprintf(fname, 1024, "%s/random_chunk%d.bin", argv[1], i);
		fp = fopen(fname, "wb");
		fwrite(chunk, elem_num * sizeof(float), 1, fp);
		fclose(fp);
	}

	mkl_free(chunk);
	return 0;
}