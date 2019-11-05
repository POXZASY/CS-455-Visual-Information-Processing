#include <Windows.h>
#include "pch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <iostream>
#include <queue>
#include <math.h>

using namespace std;
using namespace cv;

Mat grayscale(Mat img) {
	Mat gray = Mat::zeros(img.rows, img.cols, CV_8U);
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			gray.at<uchar>(y, x) = round((img.at<Vec3b>(y, x)[0] + img.at<Vec3b>(y, x)[1] + img.at<Vec3b>(y, x)[2]) / 3);
		}
	}
	return gray;
}

Mat toBinary(Mat img, int threshold, int upper) {
	Mat bin = Mat::zeros(img.rows, img.cols, CV_8U);
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			int val = (int)img.at<uchar>(y, x);
			if (val >= threshold && val < upper) bin.at<uchar>(y, x)= 255;
			else bin.at<uchar>(y, x) = 0;
		}
	}
	return bin;
}

Mat toBinaryColor()

//http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth07.pdf

//https://en.wikipedia.org/wiki/Erosion_(morphology)

//https://en.wikipedia.org/wiki/Dilation_(morphology)

//255 for yes, 0 for no (assume center is origin. this implies odd # rows / cols)
Mat erosionMatrix(int rows, int columns){
	Mat erosion = Mat::zeros(rows, columns, CV_8U);
	//fill the entire thing with 255 (this can be changed later for a different effect)
	for (int x = 0; x < erosion.cols; x++) {
		for (int y = 0; y < erosion.rows; y++) {
			erosion.at<uchar>(y, x) = 255;
		}
	}
	return erosion;
}

//255 for yes, 0 for no (assume center is origin. this implies odd # rows / cols)
Mat dilationMatrix(int rows, int columns) {
	Mat dilation = Mat::zeros(rows, columns, CV_8U);
	//fill the entire thing with 255 (this can be changed later for a different effect)
	for (int x = 0; x < dilation.cols; x++) {
		for (int y = 0; y < dilation.rows; y++) {
			dilation.at<uchar>(y, x) = 255;
		}
	}
	return dilation;
}

//using a binary matrix, assuming the erosion/dilation matrix is odd x odd dimension
//Note: while this function performs either erosion or dilation, the matrix is labeled for erosion throughout
Mat erosionOrDilation(Mat bin, Mat erosion, bool erode) {
	Mat erodedMat = bin.clone();
	//iterate through each pixel in the binary image
	for (int x = 0; x < bin.cols; x++) {
		for (int y = 0; y < bin.rows; y++) {
			//check if in bounds
			//check x bounds
			int distx = (erosion.cols - 1) / 2;
			if (x < distx || x > bin.cols - 1 - distx) continue;
			//check y bounds
			int disty = (erosion.rows - 1) / 2;
			if (y < disty || y > bin.rows - 1 - disty) continue;
			//if operation is erosion
			if (erode == true) {
				int dilationcount = 0;
				for (int i = 0; i < erosion.cols; i++) {
					for (int j = 0; j < erosion.rows; j++) {
						if (bin.at<uchar>(y - disty + j, x - distx + i) != erosion.at<uchar>(j, i)) {
							erodedMat.at<uchar>(y, x) = 0;
						}
					}
				}
			}
			//if operation is dilation
			else {
				if (bin.at<uchar>(y, x) == 255) {
					for (int i = 0; i < erosion.cols; i++) {
						for (int j = 0; j < erosion.rows; j++) {
							erodedMat.at<uchar>(y - disty + j, x - distx + i) = 255;
						}
					}
				}
			}
		}
	}
	return erodedMat;
}


Mat performErosion(Mat bin, Mat erosion) {
	return erosionOrDilation(bin, erosion, true);
}



Mat performDilation(Mat bin, Mat dilation) {
	return erosionOrDilation(bin, dilation, false);
}



////////////////////////////////////////////
/////////REGION COLORING////////////////////
////////////////////////////////////////////

int label = 1;


void searchRegion(vector<vector<pair<int, int>>> &map, queue<pair<int, int>> &pointqueue, Mat binary) {
	pair<int, int> point = pointqueue.front();
	pointqueue.pop();
	int x = point.first;
	int y = point.second;
	//go left
	if (x > 0) {
		pair<int, int> temppoint = map[x - 1][y];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x - 1][y] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x - 1, y));
		}
	}
	//go right
	if (x < binary.cols - 1) {
		pair<int, int> temppoint = map[x + 1][y];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x + 1][y] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x + 1, y));
		}
	}
	//go up
	if (y > 0) {
		pair<int, int> temppoint = map[x][y - 1];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x][y - 1] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x, y - 1));
		}
	}
	//go down
	if (y < binary.rows - 1) {
		pair<int, int> temppoint = map[x][y + 1];
		if (temppoint.first != 0 && temppoint.second == 0) { //foreground pixel, not already labeled
			map[x][y + 1] = make_pair(temppoint.first, label); //label pixel
			pointqueue.push(make_pair(x, y + 1));
		}
	}
}

vector<vector<pair<int, int>>> regionMap(Mat binary) {
	//implementing first algorithm on https://en.wikipedia.org/wiki/Connected-component_labeling
	vector<vector<pair<int, int>>> map(binary.cols, vector<pair<int, int>>(binary.rows));
	//initialize all values in map
	for (int x = 0; x < binary.cols; x++) {
		for (int y = 0; y < binary.rows; y++) {
			map[x][y] = make_pair(binary.at<uchar>(y, x), 0); //first value is value in binary image (0 or 255), second value is "label", default 0
		}
	}

	queue<pair<int, int>> pointqueue; //locations (x,y)

	for (int x = 0; x < binary.cols; x++) {
		for (int y = 0; y < binary.rows; y++) {
			//if point hasn't been accessed yet
			if (map[x][y].first > 0 && map[x][y].second == 0) {
				map[x][y] = make_pair(map[x][y].first, label);
				pointqueue.push(make_pair(x, y));
				while (pointqueue.size() > 0) {
					searchRegion(map, pointqueue, binary);
				}
				label++;
			}
		}
	}
	return map;
}

struct sorter {
	inline bool operator()(const vector<pair<int, int>> vec1, const vector<pair<int, int>> vec2) {
		return vec1.size() > vec2.size();
	}
};

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
				regions[temppoint.second - 1].push_back(make_pair(x, y));
			}
		}
	}

	//sort regions by size
	sort(regions.begin(), regions.end(), sorter());

	/////////////////////////////////////////////////////////////////////
	///ASSIGNMENT 4 SPECIFIC: REMOVE ALL REGIONS SMALLER THAN N PIXELS///
	/////////////////////////////////////////////////////////////////////

	int minsize = 15;
	bool successfulcycle = false;
	while (!successfulcycle) {
		successfulcycle = true;
		for (int i = 0; i < regions.size(); i++) {
			if (regions[i].size() < 15) {
				successfulcycle = false;
				regions.erase(regions.begin() + i);
				break;
			}
		}
	}
	

	Mat coloredRegions = Mat::zeros(binary.rows, binary.cols, 16); //type 16 is of given figure
	cout << "Number of regions: " << regions.size() << endl;


	for (int j = 0; j < regions.size(); j++) {
		for (int i = 0; i < regions[j].size(); i++) {
			pair<int, int> point = regions[j][i];
			if (j == 0) coloredRegions.at<Vec3b>(point.second, point.first)[2] = 255; //set largest to red
			else if (j < regions.size() - 1) coloredRegions.at<Vec3b>(point.second, point.first)[1] = 255; //set middle shapes to green
			else if (j == regions.size() - 1) coloredRegions.at<Vec3b>(point.second, point.first)[0] = 255;
		}
	}


	return coloredRegions;
}

///////////////////////////////////////////

int main(){
	int numscenes = 8;
	int scenecount = 4;
	//images
	string imgnames[] = {"TestImage-even-width.bmp", "Picture3-small.bmp"};
	Mat fig1 = imread(imgnames[0], CV_8U); // Read the file
	Mat fig2 = imread(imgnames[1], CV_32S); // Read the file
	Mat binary = toBinary(fig1, 200, 256);
	//perform erosion
	Mat eroMat = erosionMatrix(5, 5);
	Mat eroded = performErosion(binary, eroMat);
	//perform opening
	Mat dilMat = dilationMatrix(9, 9);
	Mat dilated = performDilation(eroded, dilMat);
	//Abnormal cells
	Mat gray = grayscale(fig2);
	Mat bincells = toBinary(gray, 220, 230);
	//perform erosion
	Mat eroMatCells = erosionMatrix(3, 3);
	Mat erodedCells = performErosion(bincells, eroMatCells);
	//perform opening
	Mat dilMatCells = dilationMatrix(3, 3);
	Mat dilatedCells = performDilation(erodedCells, dilMatCells);
	while (scenecount < numscenes) {
		if (scenecount == 0) {
			imshow("Binary Image", binary);
		}
		if (scenecount == 1) {
			imshow("Image after Erosion", eroded);
		}
		if (scenecount == 2) {
			imshow("Image after Opening", dilated);
		}
		if (scenecount == 3) {
			imshow("Colored Regions", regionSizing(dilated));
		}
		if (scenecount == 4) {
			imshow("Abnormal Cells", fig2);
		}
		if (scenecount == 5) {
			imshow("Grayscale Abnormal Cells", gray);
		}
		if (scenecount == 6) {
			imshow("Binary Abnormal Cells", bincells);
		}
		if (scenecount == 7) {
			imshow("Closure on Abnormal Cells", dilatedCells);
		}
		waitKey();
		scenecount++;
	}
	return 0;
}