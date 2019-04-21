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
//#include "laplacianZC.h"
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
	
	cv::Mat image1 = cv::imread("d:/image/images/church01.jpg");
	cv::Mat image2 = cv::imread("d:/image/images/church02.jpg");
	if (image1.empty()) {
		cerr <<" 图像读取失败" << endl;
	}
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	cv::Ptr<cv::FeatureDetector> ptrDetector; //泛型检测器指针
	ptrDetector = cv::FastFeatureDetector::create(80);

	//检测关键点
	ptrDetector->detect(image1, keypoints1);
	ptrDetector->detect(image2, keypoints2);

	//定义正方形的邻域
	const int nsize(11); //尺寸
	cv::Rect neighborhood(0, 0, nsize, nsize);
	cv::Mat patch1;
	cv::Mat patch2;

	cv::Mat result;
	std::vector<cv::DMatch> matches;

	for (int i = 0; i < keypoints1.size(); i++) {
		neighborhood.x = keypoints1[i].pt.x - nsize / 2;
		neighborhood.y = keypoints1[i].pt.y - nsize / 2;
		if (neighborhood.x<0 || neighborhood.y<0 || neighborhood.x + nsize>image1.cols || neighborhood.y + nsize>image1.rows) {
			continue;
		}
		patch1 = image1(neighborhood);
		cv::DMatch bestMatch;

		//for all keypoints in image 2
		for (int j = 0; j<keypoints2.size(); j++) {

			// define image patch
			neighborhood.x = keypoints2[j].pt.x - nsize / 2;
			neighborhood.y = keypoints2[j].pt.y - nsize / 2;

			// if neighborhood of points outside image, then continue with next point
			if (neighborhood.x<0 || neighborhood.y<0 ||
				neighborhood.x + nsize >= image2.cols || neighborhood.y + nsize >= image2.rows)
				continue;

			// patch in image 2
			patch2 = image2(neighborhood);

			// match the two patches
			cv::matchTemplate(patch1, patch2, result, cv::TM_SQDIFF);

			// check if it is a best match
			if (result.at<float>(0, 0) < bestMatch.distance) {

				bestMatch.distance = result.at<float>(0, 0);
				bestMatch.queryIdx = i;
				bestMatch.trainIdx = j;
			}
		}
		// add the best match
		matches.push_back(bestMatch);
	}
	std::cout << "Number of matches: " << matches.size() << std::endl;

	// extract the 50 best matches
	std::nth_element(matches.begin(), matches.begin() + 50, matches.end());
	matches.erase(matches.begin() + 50, matches.end());

	std::cout << "Number of matches (after): " << matches.size() << std::endl;

	// Draw the matching results
	cv::Mat matchImage;
	cv::drawMatches(image1, keypoints1, // first image
		image2, keypoints2, // second image
		matches,     // vector of matches
		matchImage,  // produced image
		cv::Scalar(255, 255, 255),  // line color
		cv::Scalar(255, 255, 255)); // point color

									// Display the image of matches
	cv::namedWindow("Matches");
	cv::imshow("Matches", matchImage);


	// define a template
	cv::Mat target(image1, cv::Rect(80, 105, 30, 30));
	// Display the template
	cv::namedWindow("Template");
	cv::imshow("Template", target);

	// define search region
	cv::Mat roi(image2,
		// here top half of the image
		cv::Rect(0, 0, image2.cols, image2.rows / 2));

	// perform template matching
	cv::matchTemplate(
		roi,    // search region
		target, // template
		result, // result
		CV_TM_SQDIFF); // similarity measure

					   // find most similar location
	double minVal, maxVal;
	cv::Point minPt, maxPt;
	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);

	// draw rectangle at most similar location
	// at minPt in this case
	cv::rectangle(roi, cv::Rect(minPt.x, minPt.y, target.cols, target.rows), 255);

	// Display the template
	cv::namedWindow("Best");
	cv::imshow("Best", image2);


	waitKey(0);
	destroyAllWindows;

	system("pause");
	return 0;
	
}