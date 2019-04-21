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
		cerr <<" Í¼Ïñ¶ÁÈ¡Ê§°Ü" << endl;
	}
	// 2. Define keypoints vector
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	// 3. Define feature detector
	// Construct the SURF feature detector object
	cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SURF::create(2000.0);
	// to test with SIFT instead of SURF 
	// cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);

	// 4. Keypoint detection
	// Detect the SURF features
	ptrFeature2D->detect(image1, keypoints1);
	ptrFeature2D->detect(image2, keypoints2);

	// Draw feature points
	cv::Mat featureImage;
	cv::drawKeypoints(image1, keypoints1, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Display the corners
	cv::namedWindow("SURF");
	cv::imshow("SURF", featureImage);

	// 5. Extract the descriptor
	cv::Mat descriptors1;
	cv::Mat descriptors2;
	ptrFeature2D->compute(image1, keypoints1, descriptors1);
	ptrFeature2D->compute(image2, keypoints2, descriptors2);

	//Construction of the matcher 
	cv::BFMatcher matcher(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	cv::Mat imageMatches;
	cv::drawMatches(image1,keypoints1, // 1st image and its keypoints
		image2,keypoints2,				// 2nd image and its keypoints
		matches,imageMatches,			// the matches,the image produced
		cv::Scalar(255,255,255),cv::Scalar(255,255,255),// color of lines, color of points
		std::vector<char>(),					// masks if any 
		cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Display the image of matches
	cv::namedWindow("SURF Matches");
	cv::imshow("SURF Matches", imageMatches);

	std::cout << "Number of matches: " << matches.size() << std::endl;
	waitKey(0);
	destroyAllWindows;

	system("pause");
	return 0;
	
}