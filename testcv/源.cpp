#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\xfeatures2d.hpp>
#include "watershed.h"
#include <opencv2\features2d\features2d.hpp>
#include "laplacianZC.h"
#include"edgedetector.h"
#include "harrisDetector.h"
using namespace std;
using namespace cv;

void colorReduce(const Mat &image, Mat &result,int div);
void colorReduce2(const Mat &image, Mat &result, int div);
void colorReduce3(const Mat &image, Mat &result, int div);
void sharpen(const Mat &image, Mat &result);

void colorReduce(const Mat &image, Mat &result, int div) {
	result = image.clone();
	int channels = result.channels();
	int rows = result.rows;
	int cols = result.cols*channels;
	for (int i = 0; i < rows; i++) {
		uchar *data = result.ptr<uchar>(i);
		//*data++ = (data[i] / div)*div + div / 2;
		for (int j = 0; j < cols; j++) {
			data[j] = (data[j] / div)*div + div / 2;
		}
	}
}

void colorReduce2(const Mat &image, Mat &result, int div) {
	result = image.clone();
	//int channels = result.channels();
	int rows = result.rows;
	int cols = result.cols;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result.at<Vec3b>(i, j)[0] = (result.at<Vec3b>(i, j)[0] / div)*div + div / 2;
			result.at<Vec3b>(i, j)[1] = (result.at<Vec3b>(i, j)[1] / div)*div + div / 2;
			result.at<Vec3b>(i, j)[2] = (result.at<Vec3b>(i, j)[2] / div)*div + div / 2;
		}
	}
}

void colorReduce3(const Mat &image, Mat &result, int div) {
	result = image.clone();
	Mat_<Vec3b>::iterator it = result.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = result.end<Vec3b>();
	for (; it != itend; it++) {
		(*it)[0] = ((*it)[0] / div)*div + div / 2;
		(*it)[1] = ((*it)[1] / div)*div + div / 2;
		(*it)[2] = ((*it)[2] / div)*div + div / 2;
	}
}


void sharpen(const Mat &image, Mat &result) {
	result.create(image.rows, image.cols, image.type());
	//result = image.clone();
	int nchannels = image.channels();
	for (int j = 1; j < image.rows - 1; j++) {
		const uchar* previous = image.ptr<const uchar>(j - 1);
		const uchar* current = image.ptr<const uchar>(j);
		const uchar* next = image.ptr<const uchar>(j + 1);
 
		uchar* output = result.ptr<uchar>(j);
		for (int i = nchannels; i < (image.cols - 1)*nchannels; i++) {
			*output++ = saturate_cast<uchar>(5 * current[i] -
				current[i - nchannels] - 
				current[i + nchannels] - previous[i] - next[i]);

		}
		result.row(0).setTo(Scalar(0));
		result.row(result.rows - 1).setTo(Scalar(0));
		result.col(0).setTo(Scalar(0));
		result.col(result.cols - 1).setTo(Scalar(0));
	}
}

int main() {
	//const int64 start = getTickCount();
	Mat image1 = imread("d:/image/images/road.jpg",0);
	Mat image2 = imread("d:/image/images/rain.jpg",0);
	if (!image1.data) {
		return  0;
	}
	if (!image2.data) {
		return 0;
	}
	
	
	cv::Mat image = cv::imread("d:/image/images/church01.jpg", 0);
	if (!image.data) {
		return 0;
	}


	cv::transpose(image, image);
	cv::flip(image, image, 0);

	std::vector<cv::KeyPoint> keypoints;
	cv::Ptr<cv::GFTTDetector> ptrGFTT = cv::GFTTDetector::create(500, 0.01, 10);
	ptrGFTT->detect(image, keypoints);

	std::vector<cv::KeyPoint>::const_iterator it = keypoints.begin();
	while (it != keypoints.end()) {
		cv::circle(image, it->pt, 3, cv::Scalar(255, 255, 255), 1);
		++it;
	}
	// Display the keypoints
	cv::namedWindow("GFTT");
	cv::imshow("GFTT", image);

	// Read input image
	image = cv::imread("d:/image/images/church01.jpg", 0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	// draw the keypoints
	cv::drawKeypoints(image,		// original image
		keypoints,					// vector of keypoints
		image,						// the resulting image
		cv::Scalar(255, 255, 255),	// color of the points
		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

												 // Display the keypoints
	cv::namedWindow("Good Features to Track Detector");
	cv::imshow("Good Features to Track Detector", image);

	image = cv::imread("d:/image/images/church01.jpg",0);
	cv::transpose(image, image);
	cv::flip(image, image, 0);
	keypoints.clear();

	cv::Ptr<cv::FastFeatureDetector> ptrFAST = cv::FastFeatureDetector::create(40);
	ptrFAST->detect(image, keypoints);
	cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	std::cout << "Number of keypoints (FAST): " << keypoints.size() << std::endl;

	cv::namedWindow("FAST");
	cv::imshow("FAST",image);

	// FAST feature without non-max suppression
	// Read input image
	image = cv::imread("d:/image/images/church01.jpg", 0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	keypoints.clear();
	// detect the keypoints
	ptrFAST->setNonmaxSuppression(false);

	ptrFAST->detect(image, keypoints);
	// draw the keypoints
	cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

	// Display the keypoints
	cv::namedWindow("FAST Features (all)");
	cv::imshow("FAST Features (all)", image);

	image = cv::imread("d:/image/images/church01.jpg",0);
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	//keypoints.clear();
	int total(100);
	int hstep(5), vstep(3);
	int hsize(image.cols / hstep), vsize(image.rows / vstep);
	int subtotal(total / (hstep*vstep));
	cv::Mat imageROI;
	std::vector<cv::KeyPoint> gridpoints;
	std::cout << "Grid of " << vstep << " by " << hstep << " each of size " << vsize << " by " << hsize << std::endl;

	ptrFAST->setThreshold(20);
	ptrFAST->setNonmaxSuppression(true);
	keypoints.clear();

	for (int i = 0; i < vstep; i++) {
		for (int j = 0; j < hstep; j++) {
			imageROI = image(cv::Rect(j*hsize, i*vsize, hsize,vsize));
			gridpoints.clear();
			ptrFAST->detect(imageROI, gridpoints);
			std::cout << "Number of FAST in grid " << i << "," << j << ": " << gridpoints.size() << std::endl;
			if (gridpoints.size() > subtotal) {
				for (auto it = gridpoints.begin(); it != gridpoints.begin() + subtotal; ++it) {
				}
			}
			auto itEnd(gridpoints.end()); 
			if (gridpoints.size() > subtotal) {
				nth_element(gridpoints.begin(), gridpoints.begin() + subtotal, gridpoints.end(),
					[](cv::KeyPoint& a, cv::KeyPoint& b) {return a.response > b.response; });
				itEnd = gridpoints.begin() + subtotal;

			}

			for (auto it = gridpoints.begin(); it != itEnd; ++it) {
				it->pt += cv::Point2f(j*hsize, i*vsize); // convert to image coordinates
				keypoints.push_back(*it);
				std::cout << "  " << it->response << std::endl;
			}
		}
	}
	
	cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	// Display the keypoints
	cv::namedWindow("FAST Features (grid)");
	cv::imshow("FAST Features (grid)", image);

	image = cv::imread("d:/image/images/church01.jpg", 0);
	cv::transpose(image, image);
	cv::flip(image, image,0);

	keypoints.clear();
	cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> ptrSURF = cv::xfeatures2d::SurfFeatureDetector::create(2000.0);
	ptrSURF->detect(image, keypoints);

	ptrSURF->detect(image, keypoints);

	cv::Mat featureImage;
	cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::namedWindow("SURF");
	cv::imshow("SURF", featureImage);
	std::cout << "Number of SURF keypoints:" << keypoints.size() << std::endl;



	waitKey(0);
	destroyAllWindows;

	system("pause");
	return 0;
	
}