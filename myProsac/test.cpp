#include <iostream>
#include <opencv2/opencv.hpp>


#include"myprosac.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  Mat img1 = imread("../c1.jpg", 0);
  Mat img2 = imread("../c2.jpg", 0);

  vector<KeyPoint> vkps1, vkps2;
  Mat desc1, desc2;
  Ptr<ORB> detector = ORB::create();
  detector->detectAndCompute(img1, Mat(), vkps1, desc1);
  detector->detectAndCompute(img2, Mat(), vkps2, desc2);

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(NORM_HAMMING);
  vector<DMatch> matches;
  matcher->match(desc1, desc2, matches);
  cout << "original matches: " << matches.size() << endl;

  Mat img_orig;
  drawMatches(img1, vkps1, img2, vkps2, matches, img_orig, Scalar(0, 255, 0),
              Scalar(0, 0, 255));

  // DMatch -> vector<int>
  vector<int> queryIdx(matches.size()), trainIdx(matches.size());
  for (size_t i = 0; i < matches.size(); i++) {
    queryIdx[i] = matches[i].queryIdx;
    trainIdx[i] = matches[i].trainIdx;
  }

  Mat H;
  vector<Point2f> pts1, pts2;
  KeyPoint::convert(vkps1, pts1, queryIdx);
  KeyPoint::convert(vkps2, pts2, trainIdx);
  int ransacReprojThres = 10;

  double sTime = (double)getTickCount();
  H = myFindH((Mat)pts1, (Mat)pts2, PROSAC, ransacReprojThres);
  double interval = ((double)getTickCount() - sTime) / getTickFrequency();
  cout << "Find homography time: " << interval << "s" << endl;

  vector<char> mask(matches.size(), 0);
  Mat pts1_t;
  perspectiveTransform(Mat(pts1), pts1_t, H);

  int goodMatches = 0;
  for (size_t i = 0; i < pts1.size(); i++) {
    if (norm(pts2[i] - pts1_t.at<Point2f>((int)i, 0)) <= ransacReprojThres) {
      mask[i] = 1;
      goodMatches++;
    }
  }
  cout << "goodMatches: " << goodMatches << endl;

  Mat img_opti;
  drawMatches(img1, vkps1, img2, vkps2, matches, img_opti, Scalar(0, 255, 0),
              Scalar(0, 0, 255), mask);

  imshow("img_orig", img_orig);
  imshow("img_opti", img_opti);
  waitKey(0);

  return 0;
}
