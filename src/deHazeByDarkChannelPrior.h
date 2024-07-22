#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <intrin.h> 
#include <opencv2/core/hal/intrin.hpp>
int deHazeByDarkChannelPrior(cv::Mat & input, cv::Mat & output);
int deHazeByDarkChannelPrior_SIMD(cv::Mat & input, cv::Mat & output);