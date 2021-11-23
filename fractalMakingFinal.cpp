#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <omp.h>
#include <mpi.h>

using namespace cv;
using namespace std;


#define MAX_ITER 1000
#define WIDTH 1000
#define HEIGHT 1000
uchar buffer[1000*1000*50];

Mat image_generated;

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

Mat setColorsOfPixel(Mat image, int row_start, int row_end) {
	int x;
	int y;
	int iter;
	int scaled_value;

	for (x = 0; x < WIDTH; x++) {
		for (y = row_start; y < row_end; y++) {
			iter = mandelbrot(x, y);

			scaled_value = scale(MAX_ITER, iter, 255, 0);
			Vec3b pixel;
			pixel.val[0] = scaled_value;
			pixel.val[1] = scaled_value;
			pixel.val[2] = scaled_value;
			image.at<Vec3b>(Point(x, y)) = pixel;
		}
	}
	return image;
}


Mat matrcv() {
	MPI_Status status;
	int count, rows, cols, type, channels, ntasks;
	int src = 1;

	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	ntasks--;

	
	for (int i = 1; i < 3; i++) {
		MPI_Recv(&buffer, sizeof(buffer), MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);

		int row_start, row_end;
		src--;

		row_start = (double)i / (double)ntasks * HEIGHT;
		row_end = ((double)i + 1.0) / (double)ntasks * HEIGHT;

		memcpy((uchar*)&rows, &buffer[0 * sizeof(int)], sizeof(int));
		memcpy((uchar*)&cols, &buffer[1 * sizeof(int)], sizeof(int));
		memcpy((uchar*)&type, &buffer[2 * sizeof(int)], sizeof(int));

	}
	
	// Make the mat
	Mat received = Mat(rows, cols, type, (uchar*)&buffer[3 * sizeof(int)]);
	return received;
}

void matsnd(const Mat& m) {
	int rows = m.rows;
	int cols = m.cols;
	int type = m.type();
	int channels = m.channels();
	int bytes = m.rows * m.cols * channels * 1;
	int dest = 0;
	int id, ntasks;
	int row_start, row_end;

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	ntasks--;
	
	if (id == 1) {
		row_start = (double)id / (double)ntasks * HEIGHT;
		row_end = ((double)id + 1.0) / (double)ntasks * HEIGHT;

		Mat new_image = setColorsOfPixel(m, row_start, row_end);

		memcpy(&buffer[3 * sizeof(int)], new_image.data, bytes);
		MPI_Send(&buffer, bytes + 3 * sizeof(int), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
	}

	if (id == 2) {
		row_start = (double)id / (double)ntasks * HEIGHT;
		row_end = ((double)id + 1.0) / (double)ntasks * HEIGHT;

		Mat new_image = setColorsOfPixel(m, row_start, row_end);

		memcpy(&buffer[3 * sizeof(int)], new_image.data, bytes);
		MPI_Send(&buffer, bytes + 3 * sizeof(int), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
	}

}


int main(int argc, char** argv)
{
	int err, np, me, source_id;
	MPI_Status status;

	Mat image = Mat::zeros(WIDTH, HEIGHT, CV_8UC3);
	//Mat image2 = Mat::zeros(WIDTH, 5, CV_8UC3);

	
	err = MPI_Init(&argc, &argv);
	if (err != MPI_SUCCESS) {
		cout<<"Couldn't start MPI.\n";
		MPI_Abort(MPI_COMM_WORLD, err);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	
	

	
	if (np < 2) {
		if (me == 0) {
			printf("You have to use exactly 2 processors to run this program\n");
		}
		MPI_Finalize();	       
		exit(0);
	}
	Mat genMat;
	
	if (me == 0) {
		genMat = matrcv();

	}
	else if(me==1 or me==2){
		matsnd(image);

	}
	MPI_Finalize();
	
	
	//Mat image_scaled;
	//cv::resize(genMat, image_scaled, cv::Size(900, 900));

	imshow("Display Window", image);
	waitKey(0);
	

	return 0;
}
