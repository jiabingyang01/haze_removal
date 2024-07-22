#include "deHazeByDarkChannelPrior.h"

using namespace cv;
using namespace std;


#define SHOW_FRAME 1

int use_simd = 1;

int mat_testOnImg(Mat & src)
{	
	Mat dst;
	if (src.rows > 300)
		resize(src, src, Size(src.cols * 300 / src.rows, 300));
	imshow("src", src);
	// auto_tune(src, src);
	int cost_time = 0;
	if (use_simd)
	{
		 cost_time = deHazeByDarkChannelPrior_SIMD(src, dst);
	}
	else
	{
		cost_time = deHazeByDarkChannelPrior(src, dst);
	}
	imshow("dst", dst);
	return cost_time;
}

void file_testOnImg(cv::String filename)
{
	Mat src = imread(filename);
	mat_testOnImg(src);
}

void writeImg(std::string in, std::string out)   //no output will display
{
	clock_t start = clock();	
	cout << "read--------->" << in << "\n";
	Mat src = imread(in), dst;
	if (src.rows > 300)
		resize(src, src, Size(src.cols * 300 / src.rows, 300));
	int cost_time = 0;
	if (use_simd)
	{
		 cost_time = deHazeByDarkChannelPrior_SIMD(src, dst);
	}
	else
	{
		cost_time = deHazeByDarkChannelPrior(src, dst);
	}		
	std::vector<int> compression_params;  
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9); 
	try {
		imwrite(out, dst, compression_params);
		cout << "write--------->" << out << "\n";
	}
	catch (std::runtime_error & e) {
		std::cout << e.what() << "\n";
	}
	cout << "##-----------------------time:" << (clock() - start)  << " ms\n";
}


void testOnMedia(std::string filename) 
{
	VideoCapture capture(filename);
#if SHOW_FRAME
	clock_t start = clock();
	int fram = 0;
	int time_sum = 0;
	int count_time = 0;
#endif

	while (true) {
		Mat frame;
		capture >> frame;
#if SHOW_FRAME
		++fram;
		++count_time;
#endif
		if (frame.empty()) {
			cout << "empty" << endl;
			break;
		}
		assert(frame.type() == CV_8UC3);
		if (frame.rows > 200)
			resize(frame, frame, Size(frame.cols * 200 / frame.rows, 200));
		
		time_sum += mat_testOnImg(frame);

		int key = waitKey(20);
		if (key == 27)
			break;
#if SHOW_FRAME
		if (clock() - start >= 1000) {
			cout << "frame: " << fram << " /s" << "\n";
			fram = 0;
			start = clock();
		}
#endif
	}
	int mean_time = time_sum / count_time;
	cout << "mean_time: " << mean_time << " ms" << endl;
}

void writeMedia(std::string in, std::string out)
{
	VideoCapture capture(in);
#if SHOW_FRAME
    clock_t start = clock();
    int fram = 0;
	int time_sum = 0;
	int count_time = 0;	
#endif

    VideoWriter writer(out, VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, Size(355, 200));
    Mat dst;
    while (true) {
        Mat frame;
        capture >> frame;
#if SHOW_FRAME
        ++fram;
		++count_time;
#endif
        if (frame.empty()) {
            cout << "empty" << endl;
            break;
        }
        assert(frame.type() == CV_8UC3);

        resize(frame, frame, Size(355, 200));
        
		int cost_time = 0;
		if (use_simd)
		{
			time_sum += deHazeByDarkChannelPrior_SIMD(frame, dst);
		}
		else
		{
			time_sum += deHazeByDarkChannelPrior(frame, dst);
		}		

        writer << dst;
        int key = waitKey(20);
        if (key == 27)
            break;
#if SHOW_FRAME
        if (clock() - start >= 1000) {
            cout << "frame: " << fram << " /s" << "\n";
            fram = 0;
            start = clock();
        }
#endif
    }
	int mean_time = time_sum / count_time;
	cout << "mean_time: " << mean_time << " ms" << endl;	
}

int main(int argc, char const *argv[])
{
	int type = 1;
	int out = 0;
	cv::String name = "cross";
	cv::String input;
	if (type){
		input = "haze_removal/input/videos/"+name+".avi";
	}
	else
	{
		input = "haze_removal/input/images/"+name+".jpg";
	}
	if (type == false) {		// "false" means for image
								// while "true" means for vedio 
		if (out)
		{
			cv::String output = "haze_removal/output/images/"+name+".jpg";	
			writeImg(input, output);
			}
		else {
			file_testOnImg(input.c_str());
		}
	}
	else {
		if (out) {
			cv::String output = "haze_removal/output/videos/"+name+".avi";
			writeMedia(input.c_str(), output.c_str());
		}
		else {
			testOnMedia(input.c_str());
		}
	}
	waitKey(0);
	return 0;
}