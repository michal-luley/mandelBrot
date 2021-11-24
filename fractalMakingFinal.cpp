#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <vector>

using namespace cv;
using namespace std;

#define MAX_ITER 1000
#define WIDTH 900
#define HEIGHT 900

float scale(float A, float A2, float Min, float Max)
{
	float A1 = 0;
	long double percentage = (A - A1) / (A1 - A2);
	return (percentage) * (Min - Max) + Min;
}

int mandelbrot(double x_orig, double y_orig) {

	double x0 = scale(x_orig, WIDTH, -2, 2);
	double y0 = scale(y_orig, HEIGHT, -2, 2);
	double x = 0.0;
	double y = 0.0;
	double xtemp = 0.0;
	int iteration = 0;
	int max_iteration = MAX_ITER;
	while (x * x + y * y <= 4.0 and iteration < max_iteration) {
		xtemp = x * x - y * y + x0;
		y = 2 * x * y + y0;
		x = xtemp;
		iteration = iteration + 1;
	}
	return iteration;
}

void setColorsOfPixel(vector <vector<uint32_t> > pole, int row_start, int row_end) {
	int x, y;
	int iter;
	int scaled_value;

	for (x = 0; x < WIDTH; x++) {
		for (y = row_start; y < row_end; y++) {
			iter = mandelbrot(x, y);
			scaled_value = scale(MAX_ITER, iter, 255, 0);
			pole[x][y] = scaled_value;
		}
	}
}

void matrcv() {
	MPI_Status status;
	int id_source, ntasks;
	int row_start, row_end, count;
	int all;

	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	
	std::vector <vector<uint32_t> >received(HEIGHT, vector<uint32_t>(WIDTH));
	ntasks--;
	all = ntasks;
	count = WIDTH * HEIGHT;
	
	//receiving image
	while (all != 0) {
		MPI_Recv(&received[0][0], count, MPI_UINT32_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		//id_source = status.MPI_SOURCE;
		//row_start = (double)id_source / (double)ntasks * HEIGHT;
		//row_end = ((double)id_source + 1.0) / (double)ntasks * HEIGHT;
		all--;
	}
	cout << "donereceiver";
}

void matsnd() {
	int count;
	int id, ntasks;
	int row_start, row_end;

	std::vector <vector<uint32_t> >original(HEIGHT, vector<uint32_t>(WIDTH));
	
	//important MPI stuff
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	id--;
	count = WIDTH * HEIGHT;
	
	//define own rows to process;
	row_start = (double)id / (double)ntasks * HEIGHT;
	row_end = ((double)id + 1.0) / (double)ntasks * HEIGHT;
	cout << "done";
	setColorsOfPixel(original, row_start, row_end);
	MPI_Send(&original[0][0], count, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{
	int err, np, me, i, j;
	MPI_Status status;

	Mat image = Mat::zeros(WIDTH, HEIGHT, CV_8UC3);

	std::vector <vector<uint32_t> >received(10000, vector<uint32_t>(10000));

	err = MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	if (err != MPI_SUCCESS) {
		cout << "Couldn't start MPI.\n";
		MPI_Abort(MPI_COMM_WORLD, err);
	}

	if (np == 1) {
		if (me == 0) {
			printf("You have to use at least 2 processors to run this program\n");
		}
		MPI_Finalize();
		exit(0);
	}
	
	if (me == 0) {
		matrcv();
	}

	else {
		matsnd();
	}

	MPI_Finalize();
	exit(0);
}
