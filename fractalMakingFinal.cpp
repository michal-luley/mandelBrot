#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;


#define MAX_ITER 1000
#define WIDTH 10000
#define HEIGHT 10000

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

Mat setColorsOfPixel(Mat image) {

	int x;
	int y;
	int iter;
	Vec3b pixel;
	int scaled_value;

	#pragma omp parallel

	int tid = omp_get_thread_num();

	for (x = 0; x < WIDTH; x++){
		for (y = 0; y < HEIGHT; y++) {
			iter = mandelbrot(x, y);

			scaled_value = scale(MAX_ITER, iter, 255, 0);
			pixel.val[0] = scaled_value;
			pixel.val[1] = scaled_value;
			pixel.val[2] = scaled_value;
			image.at<Vec3b>(Point(x, y)) = pixel;

		}
	}

	

	return image;
}
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		//cv::resize(image_generated, image_generated, cv::Size(400, 400));
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}

int main()
{
	cout << "hello";
	Mat image = Mat::zeros(WIDTH, HEIGHT, CV_8UC3);

	Mat image_scaled;
	image_generated = setColorsOfPixel(image);

	cv::resize(image_generated, image_scaled, cv::Size(900, 900));

	imshow("Display Window", image_generated);
	waitKey(0);
	
	
	return 0;
}
