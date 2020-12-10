// client.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <CaptureSender.h>
#include <ycapture.h>
#include <vector>
#include <opencv2/core.hpp>    
#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include "faceswap.h"


using namespace cv;

Mat faceSwap(Mat frame, CascadeClassifier& face_cascade, Ptr<FacemarkLBF> facemark, Mat face_img,
	vector<vector<Point2f>> mask_shape, vector<Rect> mask_faces, bool& detected)
{
	Mat frame_gray;
	vector<Rect> faces;
	vector< vector<Point2f> > shape;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces);

	int count = (int)faces.size();

	//std::cout << count << std::endl;
	facemark->getFaces(frame_gray, faces);
	if (faces.empty()) {
		detected = false;
		return frame;
	}
	facemark->fit(frame_gray, faces, shape);

	Mat img1Warped = frame.clone();

	unsigned long numswaps = (unsigned long)min((unsigned long)shape.size(), (unsigned long)mask_shape.size());

	for (unsigned long z = 0; z < numswaps; z++) {
		vector<Point2f> points2 = shape[z];
		vector<Point2f> points1 = mask_shape[z];
		face_img.convertTo(face_img, CV_32F);
		img1Warped.convertTo(img1Warped, CV_32F);
		// Find convex hull
		vector<Point2f> boundary_image1;
		vector<Point2f> boundary_image2;
		vector<int> index;
		convexHull(Mat(points2), index, false, false);
		for (size_t i = 0; i < index.size(); i++)
		{
			boundary_image1.push_back(points1[index[i]]);
			boundary_image2.push_back(points2[index[i]]);
		}
		// Triangulation for points on the convex hull
		vector< vector<int> > triangles;
		Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
		divideIntoTriangles(rect, boundary_image2, triangles);
		// Apply affine transformation to Delaunay triangles
		for (size_t i = 0; i < triangles.size(); i++)
		{
			vector<Point2f> triangle1, triangle2;
			// Get points for img1, img2 corresponding to the triangles
			for (int j = 0; j < 3; j++)
			{
				triangle1.push_back(boundary_image1[triangles[i][j]]);
				triangle2.push_back(boundary_image2[triangles[i][j]]);
			}
			warpTriangle(face_img, img1Warped, triangle1, triangle2);
		}
		// Calculate mask
		vector<Point> hull;
		for (size_t i = 0; i < boundary_image2.size(); i++)
		{
			Point pt((int)boundary_image2[i].x, (int)boundary_image2[i].y);
			hull.push_back(pt);
		}
		Mat mask = Mat::zeros(frame.rows, frame.cols, frame.depth());
		fillConvexPoly(mask, &hull[0], (int)hull.size(), Scalar(255, 255, 255));
		// Clone seamlessly.
		Rect r = boundingRect(boundary_image2);
		Point center = (r.tl() + r.br()) / 2;
		Mat output;
		img1Warped.convertTo(img1Warped, CV_8UC3);
		seamlessClone(img1Warped, frame, mask, center, output, NORMAL_CLONE);
		//imshow("Face_Swapped", output);
		detected = true;
		return output;
	}
	return frame;
}

// メイン関数
int _tmain(int argc, _TCHAR* argv[])
{
	// カメラの準備
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return -1;
	}
	int width = cap.get(CAP_PROP_FRAME_WIDTH);	// 幅
	int height = cap.get(CAP_PROP_FRAME_HEIGHT);	// 高さ
	unsigned long fps = int(cap.get(CAP_PROP_FPS)); // fps
	Mat frame; //取得したフレーム
	Mat sendFrame;
	// 1回読み込む
	cap.read(frame);

	// 各種オブジェクトの用意
	CaptureSender sender(CS_SHARED_PATH, CS_EVENT_WRITE, CS_EVENT_READ);
	unsigned char* imageBuf = new unsigned char[width * height * 3];
	unsigned long avgTimePF = fps;
	unsigned long counter = 0;

	// cascadeの準備
	CascadeClassifier face_cascade;
	String face_cascade_name = "D:/lib/opencv-3.4.12/data/haarcascades/haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name))
	{
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	//model読み込み
	String model_file_path = "D:/work/other/ycapture-src-0.1.1/ycapture/x64/Release/lbfmodel.yaml";
	auto facemark = prepare(face_cascade, model_file_path);
	vector<Rect> mask_faces;
	vector<vector<Point2f> > mask_shape;

	//swap用 顔画像
	auto face_mask = imread("D:/work/other/ycapture-src-0.1.1/ycapture/x64/Release/face_mask.jpg");
	Mat grayface_mask;
	cvtColor(face_mask, grayface_mask, cv::COLOR_BGR2GRAY);
	equalizeHist(grayface_mask, grayface_mask);

	try {
		face_cascade.detectMultiScale(grayface_mask, mask_faces);
		facemark->getFaces(grayface_mask, mask_faces);
		if (!mask_faces.empty()) {
			facemark->fit(grayface_mask, mask_faces, mask_shape);
	
			////描画 顔全体
			//for (int i = 0; i < mask_faces.size(); ++i) {
			//	cv::rectangle(face_mask, mask_faces[i], cv::Scalar(0, 0, 255), 2);
			//}

			////描画 顔特徴点
			//for (int j = 0; j < mask_shape[0].size(); j++) {
			//	//std::cout << "(" << shape[0][j].x << "," << shape[0][j].y << ")" << std::endl;
			//	cv::circle(face_mask, cv::Point(mask_shape[0][j].x, mask_shape[0][j].y), 2, cv::Scalar(0, 255, 0), -1);
			//}
		}
	}
	catch (cv::Exception & e)
	{
		// 例外をキャッチしたらエラーメッセージを表示
		std::cerr << e.what() << std::endl;
		return -1;
	}



	while (cap.read(frame))
	{
		if (frame.empty()) {
			continue;
		}

		// ここに画像処理を書く
		try {
			bool detected;
			auto p_frame = faceSwap(frame, face_cascade, facemark, face_mask,  mask_shape, mask_faces,detected);
			if (!detected) {
				//検出できていないときはスキップ
				continue;
			}
			imshow("p_frame", p_frame);//画像を表示．
			flip(p_frame, sendFrame, 0);
			const int key = waitKey(30);

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int offs = (x + y * width) * 3;
					int b = sendFrame.at<Vec3b>(y, x)[0];
					int g = sendFrame.at<Vec3b>(y, x)[1];
					int r = sendFrame.at<Vec3b>(y, x)[2];
					imageBuf[offs + 0] = r;
					imageBuf[offs + 1] = g;
					imageBuf[offs + 2] = b;
				}
			}

			if (key == 'q'/*113*/)//qボタンが押されたとき
			{
				break;//whileループから抜ける．
			}
			else if (key == 's'/*115*/)//sが押されたとき
			{
				//フレーム画像を保存する．
				imwrite("img.png", frame);
			}
			else {
				// 送信する
				HRESULT hr = sender.Send(counter * avgTimePF, width, height, imageBuf);
				if (FAILED(hr)) {
					// エラー: 続行しない
					fprintf(stderr, "Error: 0x%08x\n", hr);
					break;
				}
				else if (hr == S_OK) {
					// 送信成功
					printf("Sent: %d\n", counter);
				}
				else {
					// フィルタ未起動。無視
					//printf("Ignored: %d\n", counter);
				}

				// 次のサンプル送信まで待機
				counter++;
				Sleep(avgTimePF);
			}
		}
		catch (cv::Exception & e)
		{
			// 例外をキャッチしたらエラーメッセージを表示
			std::cerr << e.what() << std::endl;
			return -1;
		}


	}

	// 後始末
	destroyAllWindows();
	delete[] imageBuf;
	imageBuf = NULL;
	return 0;
}

