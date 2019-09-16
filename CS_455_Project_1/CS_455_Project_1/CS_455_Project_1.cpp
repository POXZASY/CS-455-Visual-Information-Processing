#include "pch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <queue>
using namespace cv;
using namespace std;

static unsigned int th = 30; //threshold

String imagelocations[] = {"figure1.bmp","figure2.bmp","figure3.bmp","figure4.bmp"};

//returns the frequency of each intensity 0 to 255
vector<int> imageValCount(Mat image) {
	vector<int> valCounts(256, 0);
	//iterate through each pixel, counting off every time a value [0,255] is found
	for (int x = 0; x < image.cols; x++) {
		for (int y = 0; y < image.rows; y++) {
			int tempval = (image.at<Vec3b>(y, x)[0] + image.at<Vec3b>(y, x)[1] + image.at<Vec3b>(y, x)[2]) / 3;
			valCounts[tempval] = valCounts[tempval] + 1;
		}
	}
	return valCounts;
}

//returns the sum of all the intensities up to n for each n 0 to 255
vector<int> imageValSums(Mat image) {
	vector<int> counts = imageValCount(image);
	vector<int> sums;
	int sum = 0;
	for (int i = 0; i < counts.size(); i++) {
		sum = sum + counts[i];
		sums.push_back(sum);
	}
	return sums;
}

Mat imageNegative(Mat fig) {
	Mat figneg = Mat::zeros(fig.rows, fig.cols, fig.type());
	//create the negative
	for (int x = 0; x < fig.cols; x++) {
		for (int y = 0; y < fig.rows; y++) {
			for (int c = 0; c < fig.channels(); c++) {
				figneg.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(255 - fig.at<Vec3b>(y, x)[c]);
			}
		}
	}
	return figneg;
}

//http://www.ece.ubc.ca/~irenek/techpaps/introip/manual02.html
Mat histogramEqualization(Mat image) {
	Mat newimg = image.clone();
	vector<int> sums = imageValSums(newimg);
	vector<int> normalizedsums;
	int numpixels = newimg.cols*newimg.rows;
	for (int i = 0; i < sums.size(); i++) {
		normalizedsums.push_back(((float)sums[i] / numpixels) * 255);
	}
	for (int x = 0; x < newimg.cols; x++) {
		for (int y = 0; y < newimg.rows; y++) {
			for (int c = 0; c < newimg.channels(); c++) {
				newimg.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(normalizedsums[newimg.at<Vec3b>(y, x)[c]]);
			}
		}
	}


	return newimg;
}



void plotHistograms(vector<Mat> images, vector<string> imagenames) {
	int histoheight = 700;
	float widthfactor = 2;
	float histowidth = 256*widthfactor;
	float heightscaler = .5;

	for (int j = 0; j < images.size(); j++) {
		Mat histogram = Mat::ones(histoheight, histowidth, images[j].type()); //256x500 matrix of 1's
		vector<int> counts = imageValCount(images[j]);
		for (int i = 0; i < 256; i++) {
			Point pt1(i*widthfactor*15.5, histoheight * 15.5);
			Point pt2((i + 1)*widthfactor*15.5, histoheight*15.5 - counts[i]);
			rectangle(histogram, pt1, pt2, Scalar(0, 255, 0), -1, 8, 4); //green rectangle
		}
		/*
		Point pt3(256, 0);
		Point pt4(512, 300);
		rectangle(histogram, pt3, pt4, Scalar(0, 255, 0), -1, 8, 4); //green rectangle
		*/
		imshow(imagenames[j], histogram);
	}
}

Mat binaryThresholding(Mat image) {
	cout << "Threshold Binary Value: " << th << endl;
	Mat binary = Mat::zeros(image.rows, image.cols, CV_8U);
	//iterate through every pixel in the image, if its at/above the threshold then set binary pixel to 1
	for (int x = 0; x < image.cols; x++) {
		for (int y = 0; y < image.rows; y++) {
			unsigned int imageval = (image.at<Vec3b>(y, x)[0] + image.at<Vec3b>(y, x)[1] + image.at<Vec3b>(y, x)[2]) / 3;
			if (imageval >= th) {
				binary.at<uchar>(y, x) = 255;
			}
			else {
				binary.at<uchar>(y, x) = 0;
			}
		}
	}
	return binary;
}

struct sorter {
	inline bool operator()(const vector<pair<int, int>> vec1, const vector<pair<int, int>> vec2) {
		return vec1.size() > vec2.size();
	}
};


int label = 1;


void searchRegion(vector<vector<pair<int, int>>> &map, queue<pair<int, int>> &pointqueue, Mat binary) {
	pair<int, int> point = pointqueue.front();
	pointqueue.pop();
	int x = point.first;
	int y = point.second;
	//go left
	if (x > 0) {
		pair<int, int> temppoint = map[x-1][y];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x-1][y] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x - 1, y));
		}
	}
	//go right
	if (x < binary.cols-1) {
		pair<int, int> temppoint = map[x + 1][y];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x + 1][y] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x + 1, y));
		}
	}
	//go up
	if (y > 0) {
		pair<int, int> temppoint = map[x][y-1];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x][y-1] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x, y-1));
		}
	}
	//go down
	if (y < binary.rows-1) {
		pair<int, int> temppoint = map[x][y+1];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x][y+1] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x, y+1));
		}
	}
}

vector<vector<pair<int, int>>> regionMap(Mat binary) {
	//implementing first algorithm on https://en.wikipedia.org/wiki/Connected-component_labeling
	vector<vector<pair<int, int>>> map(binary.cols, vector<pair<int, int>>(binary.rows));
	//initialize all values in map
	for (int x = 0; x < binary.cols; x++) {
		for (int y = 0; y < binary.rows; y++) {
			map[x][y] = make_pair(binary.at<uchar>(y,x), 0); //first value is value in binary image (0 or 255), second value is "label", default 0
		}
	}
	
	queue<pair<int, int>> pointqueue; //locations (x,y)
	
	for (int x = 0; x < binary.cols; x++) {
		for (int y = 0; y < binary.rows; y++) {
			//if point hasn't been accessed yet
			if (map[x][y].first > 0 && map[x][y].second == 0) {
				cout << x <<","<<y<<" "<<label << endl;
				map[x][y] = make_pair(map[x][y].first, label);
				pointqueue.push(make_pair(x,y));
				while (pointqueue.size() > 0) {
					searchRegion(map, pointqueue, binary);
				}
				label++;
			}
		}
	}
	return map;
}

Mat regionSizing(Mat binary) {
	//want to get a list of regions with all of their points
	vector<vector<pair<int, int>>> regions;
	int numinitializedregions = 0; //each region encountered will be in the order they were labeled in the map

	//populate regions
	vector<vector<pair<int, int>>> map = regionMap(binary);

	for (int x = 0; x < binary.cols; x++) {
		for (int y = 0; y < binary.rows; y++) {
			pair<int, int> temppoint = map[x][y];
			if (temppoint.second > 0) {
				if (numinitializedregions < temppoint.second) {
					vector<pair<int, int>> tempvec;
					tempvec.push_back(make_pair(x, y));
					regions.push_back(tempvec);
					numinitializedregions++;
				}
				regions[temppoint.second-1].push_back(make_pair(x, y));
			}
		}
	}

	//sort regions by size
	sort(regions.begin(), regions.end(), sorter());
	   
	Mat coloredRegions = Mat::zeros(binary.rows, binary.cols, 16); //type 16 is of given figure
	
	cout << regions.size() << endl;

	for (int j = 0; j < regions.size(); j++) {
		for (int i = 0; i < regions[j].size(); i++) {
			pair<int, int> point = regions[j][i];
			if(j==0) coloredRegions.at<Vec3b>(point.second, point.first)[2] = 255; //set largest to red
			else if(j<regions.size()-1) coloredRegions.at<Vec3b>(point.second, point.first)[1] = 255; //set middle shapes to green
			else if(j==regions.size()-1) coloredRegions.at<Vec3b>(point.second, point.first)[0] = 255;
		}
	}
	

	return coloredRegions;
}






int main(int argc, char** argv){
	int imgcount = 0;
	while(imgcount < 4){
		//reading in the image
		String figureloc = imagelocations[imgcount];
		Mat fig = imread(figureloc, IMREAD_COLOR); // Read the file
		if (fig.empty()) {
			cout << "Could not find or open the image." << endl;
			return -1;
		}
		//display general image 1 - 4
		imshow(figureloc, fig);
		
		//creating negative of image 1, displaying histogram, increase quality with equalization
		//https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
		
		if (imgcount == 0) {
			//display
			Mat figneg = imageNegative(fig);
			imshow("Negative", figneg);
			//equalize the histograms
			//constuct intial histogram fpr fig 1
			Mat equalized = histogramEqualization(fig);
			imshow("Figure 1 with Histogram Equalization", equalized);
			vector<Mat> images{ fig, figneg, equalized };
			vector<string> names{ "Histogram of Figure 1", "Histogram of Negative of Figure 1" ,"Histogram of Figure 1 After Equalization" };
			plotHistograms(images, names);
		}
		if (imgcount == 1) {
			Mat equalized = histogramEqualization(fig);
			imshow("Figure 2 with Histogram Equalization", equalized);
			vector<Mat> images{ fig, equalized };
			vector<string> names{ "Histogram of Figure 2", "Histogram of Figure 2 After Equalization" };
			plotHistograms(images, names);
		}
		if (imgcount == 2) {
			vector<Mat> images{ fig };
			vector<string> names{ "Histogram of Figure 3" };
			plotHistograms(images, names);

			//create binary image
			Mat binary = binaryThresholding(fig);
			imshow("Binary of Figure 3", binary);
			Mat regions = regionSizing(binary);
			imshow("Figure 3 Regions by Size", regions);
		}
		
		if (imgcount == 3) {
			label = 1;
			th = 40;
			vector<Mat> images{ fig };
			vector<string> names{ "Histogram of Figure 4" };
			plotHistograms(images, names);

			//create binary image
			Mat binary = binaryThresholding(fig);
			imshow("Binary of Figure 4", binary);
			Mat regions = regionSizing(binary);
			imshow("Figure 4 Regions by Size", regions);
		}





		
		
		waitKey();
		imgcount++;
	}
	return 0;
}