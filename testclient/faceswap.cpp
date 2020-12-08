#include "stdafx.h"
#include <iostream>
#include "faceswap.h"
using namespace cv;
using namespace cv::face;
using namespace std;

static bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
	Mat gray;

	if (image.channels() > 1)
		cvtColor(image, gray, COLOR_BGR2GRAY);
	else
		gray = image.getMat().clone();

	equalizeHist(gray, gray);

	std::vector<Rect> faces_;
	face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
	Mat(faces_).copyTo(faces);
	return true;
}

void divideIntoTriangles(Rect rect, vector<Point2f> &points, vector<vector<int>> &delaunayTri);
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &triangle1, vector<Point2f> &triangle2);

//Divide the face into triangles for warping
void divideIntoTriangles(Rect rect, vector<Point2f> &points, vector<vector<int>> &Tri)
{

	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);
	// Insert points into subdiv
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
		subdiv.insert(*it);
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point2f> pt(3);
	vector<int> ind(3);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f triangle = triangleList[i];
		pt[0] = Point2f(triangle[0], triangle[1]);
		pt[1] = Point2f(triangle[2], triangle[3]);
		pt[2] = Point2f(triangle[4], triangle[5]);
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			for (int j = 0; j < 3; j++)
				for (size_t k = 0; k < points.size(); k++)
					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = (int)k;
			Tri.push_back(ind);
		}
	}
}
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &triangle1, vector<Point2f> &triangle2)
{
	Rect rectangle1 = boundingRect(triangle1);
	Rect rectangle2 = boundingRect(triangle2);
	// Offset points by left top corner of the respective rectangles
	vector<Point2f> triangle1Rect, triangle2Rect;
	vector<Point> triangle2RectInt;
	for (int i = 0; i < 3; i++)
	{
		triangle1Rect.push_back(Point2f(triangle1[i].x - rectangle1.x, triangle1[i].y - rectangle1.y));
		triangle2Rect.push_back(Point2f(triangle2[i].x - rectangle2.x, triangle2[i].y - rectangle2.y));
		triangle2RectInt.push_back(Point((int)(triangle2[i].x - rectangle2.x), (int)(triangle2[i].y - rectangle2.y))); // for fillConvexPoly
	}
	// Get mask by filling triangle
	Mat mask = Mat::zeros(rectangle2.height, rectangle2.width, CV_32FC3);
	fillConvexPoly(mask, triangle2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
	// Apply warpImage to small rectangular patches
	Mat img1Rect;
	img1(rectangle1).copyTo(img1Rect);
	Mat img2Rect = Mat::zeros(rectangle2.height, rectangle2.width, img1Rect.type());
	Mat warp_mat = getAffineTransform(triangle1Rect, triangle2Rect);
	warpAffine(img1Rect, img2Rect, warp_mat, img2Rect.size(), INTER_LINEAR, BORDER_REFLECT_101);
	multiply(img2Rect, mask, img2Rect);
	multiply(img2(rectangle2), Scalar(1.0, 1.0, 1.0) - mask, img2(rectangle2));
	img2(rectangle2) = img2(rectangle2) + img2Rect;
}

Ptr<FacemarkLBF> prepare(CascadeClassifier &face_cascade, String modelfile_name)
{
	//FacemarkLBF::Params params;
	Ptr<FacemarkLBF> facemark = FacemarkLBF::create();
	facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);
	facemark->loadModel(modelfile_name);
	return facemark;
}
