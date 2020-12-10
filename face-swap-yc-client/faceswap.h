#pragma once

#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp" // seamlessClone()
#include <iostream>
using namespace cv;
using namespace cv::face;
using namespace std;

static bool myDetector(InputArray image, OutputArray faces, CascadeClassifier* face_cascade);

void divideIntoTriangles(Rect rect, vector<Point2f>& points, vector< vector<int> >& delaunayTri);
void warpTriangle(Mat& img1, Mat& img2, vector<Point2f>& triangle1, vector<Point2f>& triangle2);

//Divide the face into triangles for warping
void divideIntoTriangles(Rect rect, vector<Point2f>& points, vector< vector<int> >& Tri);

void warpTriangle(Mat& img1, Mat& img2, vector<Point2f>& triangle1, vector<Point2f>& triangle2);
Ptr<FacemarkLBF> prepare(CascadeClassifier& face_cascade, String modelfile_name);

int swapMain(int argc, char** argv);