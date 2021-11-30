#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;
using namespace std::chrono;


#define MPI_ON 0
#define OPENMP_ON 0
int number_of_threads = 1;


#define MAX_ITER 1000
#define WIDTH 1000
#define HEIGHT 1000


Mat image_main;

float scale(float A, float A2, float Min, float Max)
{
	float A1 = 0;
	long double percentage = (A - A1) / (A1 - A2);
	return (percentage) * (Min - Max) + Min;
}

int mandelbrot(double x_orig, double y_orig) {

	double x0 = scale(x_orig, WIDTH, -2, 2);
	double y0 = scale(y_orig, HEIGHT, -2, 2);
	double x = 0.0, y = 0.0, xtemp = 0.0;
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

void setColorsOfPixel(vector <uint32_t> & pole, int row_start, int row_end) {
	int iter;
	int scaled_value;

	if (OPENMP_ON) {
		omp_set_num_threads(number_of_threads);
		#pragma omp parallel shared(pole) private(iter, scaled_value) 
		{
		#pragma omp for
			for (int x = row_start; x < row_end; x++) {
				for (int y = 0; y < WIDTH; y++) {
					iter = mandelbrot(x, y);
					scaled_value = scale(iter, MAX_ITER, 255, 0);
					pole[(x - row_start) * WIDTH + y] = scaled_value;
				}
			}
		}
	}

	else {
		for (int x = row_start; x < row_end; x++) {
			for (int y = 0; y < WIDTH; y++) {
				iter = mandelbrot(x, y);
				scaled_value = scale(iter, MAX_ITER, 255, 0);
				pole[(x - row_start) * WIDTH + y] = scaled_value;
			}
		}
	}
}

void transform2Mat(vector<uint32_t> received_helper, Mat image, int rows_sum, int id, int rows_per_send) {
	if (OPENMP_ON) {
		omp_set_num_threads(number_of_threads);
		#pragma omp parallel shared(image)
		{
		#pragma omp for
			for (int i = 0; i < rows_sum; i++) {
				for (int j = 0; j < WIDTH; j++) {
					for (int k = 0; k < 3; k++) {
						image.at<Vec3b>(Point(i + id * rows_per_send, j)).val[k] = received_helper[i * WIDTH + j];
					}
				}
			}
		}
	}
	for (int i = 0; i < rows_sum; i++) {
		for (int j = 0; j < WIDTH; j++) {
			for (int k = 0; k < 3; k++) {
				image.at<Vec3b>(Point(i + id * rows_per_send, j)).val[k] = received_helper[i * WIDTH + j];
			}
		}
	}
}

Mat matrcv() {
	//esentials
	MPI_Request recv_req;
	MPI_Status status;
	int id_source, ntasks, row_start, row_end, count;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	int number_of_rows_per_send = HEIGHT / ntasks;
	std:: vector <vector<uint32_t> >received(ntasks-1, vector<uint32_t> (number_of_rows_per_send*WIDTH));
	Mat image = Mat::zeros(WIDTH, HEIGHT, CV_8UC3);
	int job_done = 0;

	//prijimanie obrazka
	for (int source_id = 0; source_id < (ntasks-1); source_id++) {
		vector<uint32_t> received_helper(number_of_rows_per_send * WIDTH);
		MPI_Irecv(&received_helper[0], number_of_rows_per_send * WIDTH, MPI_UINT32_T, source_id+1, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);

		if (job_done == 0) {
			row_start = number_of_rows_per_send * (ntasks - 1);
			row_end = HEIGHT;
			
			int rows_local = row_end - row_start;
			cout << rows_local << endl;
			
			vector<uint32_t> received_helper_local(rows_local * WIDTH);

			setColorsOfPixel(received_helper_local, row_start, row_end);
			
			transform2Mat(received_helper_local, image, rows_local, (ntasks - 1), number_of_rows_per_send);
			
			job_done = 1;
		}
		
		MPI_Wait(&recv_req, &status);
		transform2Mat(received_helper, image, number_of_rows_per_send, source_id, number_of_rows_per_send);
	}
	if (ntasks == 1) {
		row_start = number_of_rows_per_send * (ntasks - 1);
		row_end = HEIGHT;
		int rows_local = row_end - row_start;

		vector<uint32_t> received_helper_local(rows_local * WIDTH);
		setColorsOfPixel(received_helper_local, row_start, row_end);
		transform2Mat(received_helper_local, image, rows_local, (ntasks - 1), number_of_rows_per_send);
	}
	cout << "donereceiver";
	return image;
}

void matsnd() {
	int count;
	int id, ntasks;
	int row_start, row_end;

	//important MPI stuff
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	id--;

	//define esentials of process;
	int number_of_rows_per_send = HEIGHT / ntasks;
	row_start = id * number_of_rows_per_send;
	row_end = (id+1) * number_of_rows_per_send;

	std::vector<uint32_t> original(number_of_rows_per_send * WIDTH);

	//computation
	setColorsOfPixel(original, row_start, row_end);

	//sending
	MPI_Send(&original[0], number_of_rows_per_send * WIDTH, MPI_UINT32_T, 0, id, MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
	Mat imageResized;
	int err, np, me, i, j, method_type;
	int stats[10];

	if (MPI_ON) {
		MPI_Status status;

		err = MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &me);
		MPI_Comm_size(MPI_COMM_WORLD, &np);

		if (err != MPI_SUCCESS) {
			cout << "Couldn't start MPI.\n";
			MPI_Abort(MPI_COMM_WORLD, err);
		}
		auto time_start = high_resolution_clock::now();

		if (me == 0) {
			image_main = matrcv();

			auto time_stop = high_resolution_clock::now();
			auto time_duration = duration_cast<seconds>(time_stop - time_start);
			cv::resize(image_main, image_main, cv::Size(900, 900));
			imshow("MandelBrot", image_main);
			waitKey(0);
		}

		else {
			matsnd();
		}

		MPI_Finalize();
	}

	else {
		Mat image = Mat::zeros(WIDTH, HEIGHT, CV_8UC3);
		std::vector<uint32_t> original(HEIGHT * WIDTH);

		auto time_start = high_resolution_clock::now();
		setColorsOfPixel(original, 0, HEIGHT);
		transform2Mat(original, image, HEIGHT, 0, HEIGHT);
		auto time_stop = high_resolution_clock::now();
		auto time_duration = duration_cast<seconds>(time_stop - time_start);
		//cv::resize(image, image, cv::Size(900, 900));
		//imshow("MandelBrot", image);
		//waitKey(0);
		//exit(0);
	}
	

}


