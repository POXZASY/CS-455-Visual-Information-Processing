#include "pch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String imagelocations[] = { "ant_gray.bmp", "basel_gray.bmp" };

#define PI 3.14159

Mat convolution(Mat image, Mat convmatrix) {
	Mat newimage = Mat::zeros(image.rows, image.cols, image.type());
	int dist = (convmatrix.cols - 1) / 2;
	//iterate through each pixel
	for (int x = 0; x < image.cols; x++) {
		for (int y = 0; y < image.rows; y++) {
			//check if convolution matrix fits (not near edge)
			if (x-dist<0||y-dist<0||x+dist>=image.cols||y+dist>=image.rows) {
				newimage.at<uchar>(y, x) = image.at<uchar>(y, x);
				continue;
			}
			//convolute for pixel
			int value = 0;
			for (int i = x - dist; i <= x + dist; i++) {
				for (int j = y - dist; j <= y + dist; j++) {
					value += image.at<uchar>(j, i)*convmatrix.at<int>(j-y+dist,i-x+dist);
				}
			}
			if (value < 0) value = 0;
			if (value > 255) value = 255;
			newimage.at<uchar>(y, x) = value;
		}
	}
	return newimage;
}

Mat unsharpMask(int scale) {
	Mat mask = Mat::zeros(3, 3, CV_32S);
	mask.at<int>(1, 0) = -1*scale;
	mask.at<int>(0, 1) = -1*scale;
	mask.at<int>(2, 1) = -1*scale;
	mask.at<int>(1, 2) = -1*scale;
	mask.at<int>(1, 1) = 5 * scale;
	return mask;
}

Mat sobelMaskX() {
	Mat mask = Mat::zeros(3, 3, CV_32S);
	mask.at<int>(0, 0) = -1;
	mask.at<int>(0, 1) = -2;
	mask.at<int>(0, 2) = -1;
	mask.at<int>(2, 0) = 1;
	mask.at<int>(2, 1) = 2;
	mask.at<int>(2, 2) = 1;
	return mask;
}
Mat sobelMaskY() {
	Mat mask = Mat::zeros(3, 3, CV_32S);
	mask.at<int>(0, 0) = -1;
	mask.at<int>(1, 0) = -2;
	mask.at<int>(2, 0) = -1;
	mask.at<int>(0, 2) = 1;
	mask.at<int>(1, 2) = 2;
	mask.at<int>(2, 2) = 1;
	return mask;
}
Mat sobel(Mat image) {
	Mat convx = convolution(image, sobelMaskX());
	Mat convy = convolution(image, sobelMaskY());
	Mat newimage = Mat::zeros(image.rows, image.cols, CV_8U);
	for (int x = 0; x < newimage.cols; x++) {
		for (int y = 0; y < newimage.rows; y++) {
			newimage.at<uchar>(y, x) = (unsigned int)floor(sqrt( pow(convx.at<uchar>(y, x),2) + pow(convy.at<uchar>(y, x),2) ));
		}
	}
	return newimage;
}


//https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
Mat LoG(Mat image, float sigma, int dimension) {
	Mat logmatrix = Mat::zeros(dimension, dimension, CV_8U);
	float coeff = -1 / (PI*pow(sigma, 4));


	//must normalize values so they sum to 1

	return convolution(image, logmatrix);
}

int main(){
	int numscenes = 4;
	int scenecount = 0;
	while(scenecount < numscenes) {
		for (String imgloc : imagelocations) {
			Mat fig = imread(imgloc, CV_8U); // Read the file
			if (fig.empty()) {
				cout << "Could not find or open the image." << endl;
				return -1;
			}
			//display general image
			if (scenecount == 0) imshow("Default " + imgloc, fig);
			//Display unsharp image
			if (scenecount == 1) {
				Mat mask = unsharpMask(1);
				Mat sharp = convolution(fig, mask);
				imshow("Unsharp Mask on " + imgloc, sharp);
			}
			if (scenecount == 2) {
				Mat sbl = sobel(fig);
				imshow("Sobel Mask on " + imgloc, sbl);
			}
		}
		waitKey();
		scenecount++;
	}



}


