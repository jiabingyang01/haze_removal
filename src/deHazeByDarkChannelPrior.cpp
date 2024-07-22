#include "deHazeByDarkChannelPrior.h"

using namespace cv;
using namespace std;


typedef struct Pixel 
{
    int x, y;
    int data;
}Pixel;

bool structCmp(const Pixel &a, const Pixel &b) 
{
    return a.data > b.data;//descending降序
}

Mat minFilter(Mat srcImage, int kernelSize);
void makeDepth32f(Mat& source, Mat& output);
void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon);
Mat getTransmission_dark(Mat& srcimg, Mat& darkimg, int *array, int windowsize);
Mat recover(Mat& srcimg, Mat& t, float *array, int windowsize);

Mat minFilter_SIMD(Mat srcImage, int kernelSize);
void guidedFilter_SIMD(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon);
Mat getTransmission_dark_SIMD(Mat& srcimg, Mat& darkimg, int *array, int windowsize);



int deHazeByDarkChannelPrior(cv::Mat & input, cv::Mat & output) 
{
    clock_t start = clock();
    // string name = "0";
    // string loc = input;
    double scale = 1.0;
    // clock_t start, finish;
    // double duration;

    // cout << "A defog program" << endl
    //     << "----------------" << endl;

    Mat image = input;
    Mat resizedImage;
    int originRows = image.rows;
    int originCols = image.cols;

    if (scale < 1.0) 
    {
        resize(image, resizedImage, Size(originCols * scale, originRows * scale));
    }
    else 
    {

        scale = 1.0;
        resizedImage = image;
    }

    int rows = resizedImage.rows;
    int cols = resizedImage.cols;
    Mat convertImage;
    resizedImage.convertTo(convertImage, CV_32FC3, 1 / 255.0, 0);
    int kernelSize =  15 < max((rows * 0.01), (cols * 0.01)) ? 15 : max((rows * 0.01), (cols * 0.01));
    //int kernelSize = 15;
    int parse = kernelSize / 2;
    Mat darkChannel(rows, cols, CV_8UC1);
    Mat normalDark(rows, cols, CV_32FC1);
    int nr = rows;
    int nl = cols;
    float b, g, r;
    start = clock();
    // cout << "generating dark channel image." << endl;
    if (resizedImage.isContinuous()) 
    {
        nl = nr * nl;
        nr = 1;
    }
    for (int i = 0; i < nr; i++) 
    {
        float min;
        const uchar* inData = resizedImage.ptr<uchar>(i);
        uchar* outData = darkChannel.ptr<uchar>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++;
            g = *inData++;
            r = *inData++;
            min = b > g ? g : b;
            min = min > r ? r : min;
            *outData++ = min;
        }
    }
    darkChannel = minFilter(darkChannel, kernelSize);

    // imshow("darkChannel", darkChannel);
    // cout << "dark channel generated." << endl;

    //estimate Airlight
    //开一个结构体数组存暗通道，再sort，取最大0.1%，利用结构体内存储的原始坐标在原图中取点
    // cout << "estimating airlight." << endl;
    rows = darkChannel.rows, cols = darkChannel.cols;
    int pixelTot = rows * cols * 0.001;
    int *A = new int[3];
    Pixel *toppixels, *allpixels;
    toppixels = new Pixel[pixelTot];
    allpixels = new Pixel[rows * cols];


    for (unsigned int r = 0; r < rows; r++) 
    {
        const uchar *data = darkChannel.ptr<uchar>(r);
        for (unsigned int c = 0; c < cols; c++) 
        {
            allpixels[r*cols + c].data = *data;
            allpixels[r*cols + c].x = r;
            allpixels[r*cols + c].y = c;
        }
    }
    std::sort(allpixels, allpixels + rows * cols, structCmp);

    memcpy(toppixels, allpixels, pixelTot * sizeof(Pixel));

    float A_r, A_g, A_b, avg, maximum = 0;
    int idx, idy, max_x, max_y;
    for (int i = 0; i < pixelTot; i++) 
    {
        idx = allpixels[i].x; idy = allpixels[i].y;
        const uchar *data = resizedImage.ptr<uchar>(idx);
        data += 3 * idy;
        A_b = *data++;
        A_g = *data++;
        A_r = *data++;
        //cout << A_r << " " << A_g << " " << A_b << endl;
        avg = (A_r + A_g + A_b) / 3.0;
        if (maximum < avg) 
        {
            maximum = avg;
            max_x = idx;
            max_y = idy;
        }
    }

    delete[] toppixels;
    delete[] allpixels;

    for (int i = 0; i < 3; i++) 
    {
        A[i] = resizedImage.at<Vec3b>(max_x, max_y)[i];
    }
    // cout << "airlight estimated as: " << A[0] << ", " << A[1] << ", " << A[2] << endl;
    //cout << endl;

    //暗通道归一化操作（除A）
    //(I / A)
    // cout << "start normalization of input image I." << endl;
    float tmp_A[3];
    tmp_A[0] = A[0] / 255.0;
    tmp_A[1] = A[1] / 255.0;
    tmp_A[2] = A[2] / 255.0;
    for (int i = 0; i < nr; i++) 
    {
        float min = 1.0;
        const float* inData = convertImage.ptr<float>(i);
        float* outData = normalDark.ptr<float>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++ / tmp_A[0];
            g = *inData++ / tmp_A[1];
            r = *inData++ / tmp_A[2];
            min = b > g ? g : b;
            min = min > r ? r : min;
            *outData++ = min;
        }
    }
    // cout << "normalization finished." << endl << "generating relative dark channel image." << endl;
    //暗通道最小滤波
    normalDark = minFilter(normalDark, kernelSize);
    // cout << "dark channel image generated." << "start estimating transmission and guided image filtering." << endl;
    // imshow("normal", normalDark);
    int kernelSizeTrans = std::max(3, kernelSize);
    //求t与将t进行导向滤波

    Mat trans = getTransmission_dark(convertImage, normalDark, A, kernelSizeTrans);
    // cout << "tansmission estimated and guided filtered." << endl;
    // imshow("filtered t", trans);
    // cout << "start recovering." << endl;
    Mat finalImage = recover(convertImage, trans, tmp_A, kernelSize);
    // cout << "recovering finished." << endl;
    Mat resizedFinal;
    if (scale < 1.0) 
    {
        resize(finalImage, resizedFinal, Size(originCols, originRows));
        // imshow("final", resizedFinal);
    }
    // else 
    // {
    //     // imshow("final", finalImage);
    // }
    // finish = clock();
    // duration = (double)(finish - start);
    // cout << "defog used " << duration << "ms time;" << endl;
    // waitKey(0);

    finalImage.convertTo(finalImage, CV_8UC3, 255);
    // imwrite("haze_removal/results/"+ name + "_refined.png", finalImage);
    // destroyAllWindows();
    // image.release();
    // resizedImage.release();
    // convertImage.release();
    // darkChannel.release();
    // trans.release();
    // finalImage.release();
    output =  finalImage;
    int cost_time = clock() - start;
	cout << "cost time: " << cost_time  << " ms\n";
    return cost_time;
}


Mat minFilter(Mat srcImage, int kernelSize) 
{
    int radius = kernelSize / 2;

    int srcType = srcImage.type();
    int targetType = 0;
    if (srcType % 8 == 0) 
    {
        targetType = 0;
    }
    else 
    {
        targetType = 5;
    }
    Mat ret(srcImage.rows, srcImage.cols, targetType);
    Mat parseImage;
    copyMakeBorder(srcImage, parseImage, radius, radius, radius, radius, BORDER_REPLICATE);
    for (unsigned int r = 0; r < srcImage.rows; r++) 
    {
        float *fOutData = ret.ptr<float>(r);
        uchar *uOutData = ret.ptr<uchar>(r);
        for (unsigned int c = 0; c < srcImage.cols; c++) 
        {
            Rect ROI(c, r, kernelSize, kernelSize);
            Mat imageROI = parseImage(ROI);
            double minValue = 0, maxValue = 0;
            Point minPt, maxPt;
            minMaxLoc(imageROI, &minValue, &maxValue, &minPt, &maxPt);
            if (!targetType) 
            {
                *uOutData++ = (uchar)minValue;
                continue;
            }
            *fOutData++ = minValue;
        }
    }
    return ret;
}





void makeDepth32f(Mat& source, Mat& output)
{
    if ((source.depth() != CV_32F) > FLT_EPSILON)
        source.convertTo(output, CV_32F);
    else
        output = source;
}

void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
    CV_Assert(radius >= 2 && epsilon > 0);
    CV_Assert(source.data != NULL && source.channels() == 1);
    CV_Assert(guided_image.channels() == 1);
    CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

    Mat guided;
    if (guided_image.data == source.data)
    {
        //make a copy
        guided_image.copyTo(guided);
    }
    else
    {
        guided = guided_image;
    }

    //将输入扩展为32位浮点型，以便以后做乘法
    Mat source_32f, guided_32f;
    makeDepth32f(source, source_32f);
    makeDepth32f(guided, guided_32f);

    //计算I*p和I*I
    Mat mat_Ip, mat_I2;
    multiply(guided_32f, source_32f, mat_Ip);
    multiply(guided_32f, guided_32f, mat_I2);

    //计算各种均值
    Mat mean_p, mean_I, mean_Ip, mean_I2;
    Size win_size(2 * radius + 1, 2 * radius + 1);
    boxFilter(source_32f, mean_p, CV_32F, win_size);
    boxFilter(guided_32f, mean_I, CV_32F, win_size);
    boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
    boxFilter(mat_I2, mean_I2, CV_32F, win_size);

    //计算Ip的协方差和I的方差
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    var_I += epsilon;

    //求a和b
    Mat a, b;
    divide(cov_Ip, var_I, a);
    b = mean_p - a.mul(mean_I);

    //对包含像素i的所有a、b做平均
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_32F, win_size);
    boxFilter(b, mean_b, CV_32F, win_size);

    //计算输出 (depth == CV_32F)
    output = mean_a.mul(guided_32f) + mean_b;
}



Mat getTransmission_dark(Mat& srcimg, Mat& darkimg, int *array, int windowsize)
{
    //t = 1 - omega * minfilt(I / A);
    float avg_A;
    //convertImage是一个CV_32FC3的图
    if (srcimg.type() % 8 == 0) {
        avg_A = (array[0] + array[1] + array[2]) / 3.0;
    }
    else {
        avg_A = (array[0] + array[1] + array[2]) / (3.0 * 255.0);
    }
    float w = 0.95;
    int radius = windowsize / 2;
    int nr = srcimg.rows, nl = srcimg.cols;
    Mat transmission(nr, nl, CV_32FC1);

    for (int k = 0; k<nr; k++) 
    {
        const float* inData = darkimg.ptr<float>(k);
        float* outData = transmission.ptr<float>(k);
        float pix[3] = { 0 };
        for (int l = 0; l < nl; l++)
        {
            *outData++ = 1.0 - w * *inData++;
        }
    }
    // imshow("t", transmission);

    Mat trans(nr, nl, CV_32FC1);
    Mat graymat(nr, nl, CV_8UC1);
    Mat graymat_32F(nr, nl, CV_32FC1);

    if (srcimg.type() % 8 != 0) 
    {
        cvtColor(srcimg, graymat_32F, cv::COLOR_BGR2GRAY);
        guidedFilter(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
    }
    else 
    {
        cvtColor(srcimg, graymat, cv::COLOR_BGR2GRAY);

        for (int i = 0; i < nr; i++) 
        {
            const uchar* inData = graymat.ptr<uchar>(i);
            float* outData = graymat_32F.ptr<float>(i);
            for (int j = 0; j < nl; j++)
                *outData++ = *inData++ / 255.0;
        }
        guidedFilter(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
    }
    return trans;
}


Mat recover(Mat& srcimg, Mat& t, float *array, int windowsize)
{
    //J(x) = (I(x) - A) / max(t(x), t0) + A;
    int radius = windowsize / 2;
    int nr = srcimg.rows, nl = srcimg.cols;
    float tnow = t.at<float>(0, 0);
    float t0 = 0.5;
    Mat finalimg = Mat::zeros(nr, nl, CV_32FC3);
    float val = 0;

    //Be aware that transmission is a grey image
    //srcImg is a color image
    //finalImg is a color image
    //Mat store color image a pixel per 3 position
    //store grey image a pixel per 1 position
    for (unsigned int r = 0; r < nr; r++) {
        const float* transPtr = t.ptr<float>(r);
        const float* srcPtr = srcimg.ptr<float>(r);
        float* outPtr = finalimg.ptr<float>(r);
        for (unsigned int c = 0; c < nl; c++) {
            //transmission image is grey, so only need 
            //to move once per calculation, using index 
            //c(a.k.a. columns) to move is enough 
            tnow = *transPtr++;
            tnow = std::max(tnow, t0);
            for (int i = 0; i < 3; i++) {
                //so to calculate every color channel per pixel
                //move the ptr once after one calculation.
                //after 3 times move, calculation for a pixel is done
                val = (*srcPtr++ - array[i]) / tnow + array[i];
                *outPtr++ = val + 10 / 255.0;
            }
        }
    }
    // cout << finalimg.size() << endl;
    return finalimg;
}



int deHazeByDarkChannelPrior_SIMD(cv::Mat & input, cv::Mat & output) 
{
    clock_t start = clock();
    // string name = "0";
    // string loc = input;
    double scale = 1.0;
    // clock_t start, finish;
    // double duration;

    // cout << "A defog program" << endl
    //     << "----------------" << endl;

    Mat image = input;
    Mat resizedImage;
    int originRows = image.rows;
    int originCols = image.cols;

    if (scale < 1.0) 
    {
        resize(image, resizedImage, Size(originCols * scale, originRows * scale));
    }
    else 
    {

        scale = 1.0;
        resizedImage = image;
    }

    int rows = resizedImage.rows;
    int cols = resizedImage.cols;
    Mat convertImage;
    resizedImage.convertTo(convertImage, CV_32FC3, 1 / 255.0, 0);
    int kernelSize =  15 < max((rows * 0.01), (cols * 0.01)) ? 15 : max((rows * 0.01), (cols * 0.01));
    //int kernelSize = 15;
    int parse = kernelSize / 2;
    Mat darkChannel(rows, cols, CV_8UC1);
    Mat normalDark(rows, cols, CV_32FC1);
    int nr = rows;
    int nl = cols;
    float b, g, r;
    start = clock();
    // cout << "generating dark channel image." << endl;
    if (resizedImage.isContinuous()) 
    {
        nl = nr * nl;
        nr = 1;
    }
    for (int i = 0; i < nr; i++) 
    {
        float min;
        const uchar* inData = resizedImage.ptr<uchar>(i);
        uchar* outData = darkChannel.ptr<uchar>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++;
            g = *inData++;
            r = *inData++;
            min = b > g ? g : b;
            min = min > r ? r : min;
            *outData++ = min;
        }
    }
    darkChannel = minFilter_SIMD(darkChannel, kernelSize);

    // imshow("darkChannel", darkChannel);
    // cout << "dark channel generated." << endl;

    //estimate Airlight
    //开一个结构体数组存暗通道，再sort，取最大0.1%，利用结构体内存储的原始坐标在原图中取点
    // cout << "estimating airlight." << endl;
    rows = darkChannel.rows, cols = darkChannel.cols;
    int pixelTot = rows * cols * 0.001;
    int *A = new int[3];
    Pixel *toppixels, *allpixels;
    toppixels = new Pixel[pixelTot];
    allpixels = new Pixel[rows * cols];


    for (unsigned int r = 0; r < rows; r++) 
    {
        const uchar *data = darkChannel.ptr<uchar>(r);
        for (unsigned int c = 0; c < cols; c++) 
        {
            allpixels[r*cols + c].data = *data;
            allpixels[r*cols + c].x = r;
            allpixels[r*cols + c].y = c;
        }
    }
    std::sort(allpixels, allpixels + rows * cols, structCmp);

    memcpy(toppixels, allpixels, pixelTot * sizeof(Pixel));

    float A_r, A_g, A_b, avg, maximum = 0;
    int idx, idy, max_x, max_y;
    for (int i = 0; i < pixelTot; i++) 
    {
        idx = allpixels[i].x; idy = allpixels[i].y;
        const uchar *data = resizedImage.ptr<uchar>(idx);
        data += 3 * idy;
        A_b = *data++;
        A_g = *data++;
        A_r = *data++;
        //cout << A_r << " " << A_g << " " << A_b << endl;
        avg = (A_r + A_g + A_b) / 3.0;
        if (maximum < avg) 
        {
            maximum = avg;
            max_x = idx;
            max_y = idy;
        }
    }

    delete[] toppixels;
    delete[] allpixels;

    for (int i = 0; i < 3; i++) 
    {
        A[i] = resizedImage.at<Vec3b>(max_x, max_y)[i];
    }
    // cout << "airlight estimated as: " << A[0] << ", " << A[1] << ", " << A[2] << endl;
    //cout << endl;

    //暗通道归一化操作（除A）
    //(I / A)
    // cout << "start normalization of input image I." << endl;
    float tmp_A[3];
    tmp_A[0] = A[0] / 255.0;
    tmp_A[1] = A[1] / 255.0;
    tmp_A[2] = A[2] / 255.0;
    for (int i = 0; i < nr; i++) 
    {
        float min = 1.0;
        const float* inData = convertImage.ptr<float>(i);
        float* outData = normalDark.ptr<float>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++ / tmp_A[0];
            g = *inData++ / tmp_A[1];
            r = *inData++ / tmp_A[2];
            min = b > g ? g : b;
            min = min > r ? r : min;
            *outData++ = min;
        }
    }
    // cout << "normalization finished." << endl << "generating relative dark channel image." << endl;
    //暗通道最小滤波
    normalDark = minFilter_SIMD(normalDark, kernelSize);
    // cout << "dark channel image generated." << "start estimating transmission and guided image filtering." << endl;
    // imshow("normal", normalDark);
    int kernelSizeTrans = std::max(3, kernelSize);
    //求t与将t进行导向滤波

    Mat trans = getTransmission_dark_SIMD(convertImage, normalDark, A, kernelSizeTrans);
    // cout << "tansmission estimated and guided filtered." << endl;
    // imshow("filtered t", trans);
    // cout << "start recovering." << endl;
    Mat finalImage = recover(convertImage, trans, tmp_A, kernelSize);
    // cout << "recovering finished." << endl;
    Mat resizedFinal;
    if (scale < 1.0) 
    {
        resize(finalImage, resizedFinal, Size(originCols, originRows));
        // imshow("final", resizedFinal);
    }
    // else 
    // {
    //     // imshow("final", finalImage);
    // }
    // finish = clock();
    // duration = (double)(finish - start);
    // cout << "defog used " << duration << "ms time;" << endl;
    // waitKey(0);

    finalImage.convertTo(finalImage, CV_8UC3, 255);
    // imwrite("haze_removal/results/"+ name + "_refined.png", finalImage);
    // destroyAllWindows();
    // image.release();
    // resizedImage.release();
    // convertImage.release();
    // darkChannel.release();
    // trans.release();
    // finalImage.release();
    output =  finalImage;
    int cost_time = clock() - start;
	cout << "cost time: " << cost_time  << " ms\n";
    return cost_time;
}



// // SSE
// Mat minFilter_SIMD(Mat srcImage, int kernelSize) 
// {
//     int radius = kernelSize / 2;

//     int srcType = srcImage.type();
//     int targetType = (srcType % 8 == 0) ? 0 : 5;

//     Mat ret(srcImage.rows, srcImage.cols, targetType);
//     Mat parseImage;
//     copyMakeBorder(srcImage, parseImage, radius, radius, radius, radius, BORDER_REPLICATE);

//     for (unsigned int r = 0; r < srcImage.rows; r++) 
//     {
//         uchar *uOutData = ret.ptr<uchar>(r);
//         float *fOutData = ret.ptr<float>(r);

//         for (unsigned int c = 0; c < srcImage.cols; c++) 
//         {
//             Rect ROI(c, r, kernelSize, kernelSize);
//             Mat imageROI = parseImage(ROI);

//             if (!targetType) 
//             {
//                 // Use SSE for 8-bit data
//                 __m128i minVal = _mm_set1_epi8(255);

//                 for (int i = 0; i < imageROI.rows; i++) 
//                 {
//                     uchar* rowPtr = imageROI.ptr<uchar>(i);
//                     for (int j = 0; j < imageROI.cols; j += 16) 
//                     {
//                         __m128i data = _mm_loadu_si128((__m128i*)(rowPtr + j));
//                         minVal = _mm_min_epu8(minVal, data);
//                     }
//                 }

//                 // Extract the minimum value from the SIMD register
//                 alignas(16) uchar minVals[16];
//                 _mm_store_si128((__m128i*)minVals, minVal);
//                 uchar minValue = 255;
//                 for (int i = 0; i < 16; i++) 
//                 {
//                     if (minVals[i] < minValue) 
//                     {
//                         minValue = minVals[i];
//                     }
//                 }

//                 *uOutData++ = minValue;
//             }
//             else 
//             {
//                 // Use normal processing for floating point data
//                 double minValue, maxValue;
//                 minMaxLoc(imageROI, &minValue, &maxValue, nullptr, nullptr);
//                 *fOutData++ = static_cast<float>(minValue);
//             }
//         }
//     }
//     return ret;
// }


// AVX
Mat minFilter_SIMD(Mat srcImage, int kernelSize) 
{
    int radius = kernelSize / 2;
    int srcType = srcImage.type();
    int targetType = (srcType % 8 == 0) ? 0 : 5;

    Mat ret(srcImage.rows, srcImage.cols, targetType);
    Mat parseImage;
    copyMakeBorder(srcImage, parseImage, radius, radius, radius, radius, BORDER_REPLICATE);

    int alignedCols = srcImage.cols - (srcImage.cols % 8); // 32 bytes / 4 bytes (float) = 8 floats

    for (int r = 0; r < srcImage.rows; r++) 
    {
        float* fOutData = ret.ptr<float>(r);
        uchar* uOutData = ret.ptr<uchar>(r);

        for (int c = 0; c < alignedCols; c += 8) 
        {
            __m256 minValues = _mm256_set1_ps(FLT_MAX);

            for (int kr = -radius; kr <= radius; kr++) 
            {
                for (int kc = -radius; kc <= radius; kc++) 
                {
                    __m256 roiValues = _mm256_loadu_ps(reinterpret_cast<const float*>(parseImage.ptr<float>(r + radius + kr) + (c + radius + kc)));
                    minValues = _mm256_min_ps(minValues, roiValues);
                }
            }

            if (targetType == 0) 
            {
                __m256i minValuesInt = _mm256_cvtps_epi32(minValues);
                __m128i minValues128Low = _mm256_extractf128_si256(minValuesInt, 0);
                __m128i minValues128High = _mm256_extractf128_si256(minValuesInt, 1);

                minValues128Low = _mm_packus_epi32(minValues128Low, minValues128High);
                minValues128Low = _mm_packus_epi16(minValues128Low, minValues128Low);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(uOutData + c), minValues128Low);
            } 
            else 
            {
                _mm256_storeu_ps(fOutData + c, minValues);
            }
        }

        for (int c = alignedCols; c < srcImage.cols; c++) 
        {
            Rect ROI(c, r, kernelSize, kernelSize);
            Mat imageROI = parseImage(ROI);
            double minValue = 0, maxValue = 0;
            Point minPt, maxPt;
            minMaxLoc(imageROI, &minValue, &maxValue, &minPt, &maxPt);

            if (targetType == 0) 
            {
                uOutData[c] = static_cast<uchar>(minValue);
            } 
            else 
            {
                fOutData[c] = static_cast<float>(minValue);
            }
        }
    }

    return ret;
}


// void guidedFilter_SIMD(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
// {
//     CV_Assert(radius >= 2 && epsilon > 0);
//     CV_Assert(source.data != NULL && source.channels() == 1);
//     CV_Assert(guided_image.channels() == 1);
//     CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

//     Mat guided;
//     if (guided_image.data == source.data)
//     {
//         //make a copy
//         guided_image.copyTo(guided);
//     }
//     else
//     {
//         guided = guided_image;
//     }

//     //将输入扩展为32位浮点型，以便以后做乘法
//     Mat source_32f, guided_32f;
//     makeDepth32f(source, source_32f);
//     makeDepth32f(guided, guided_32f);

//     //计算I*p和I*I
//     Mat mat_Ip, mat_I2;
//     multiply(guided_32f, source_32f, mat_Ip);
//     multiply(guided_32f, guided_32f, mat_I2);

//     //计算各种均值
//     Mat mean_p, mean_I, mean_Ip, mean_I2;
//     Size win_size(2 * radius + 1, 2 * radius + 1);
//     boxFilter(source_32f, mean_p, CV_32F, win_size);
//     boxFilter(guided_32f, mean_I, CV_32F, win_size);
//     boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
//     boxFilter(mat_I2, mean_I2, CV_32F, win_size);

//     //计算Ip的协方差和I的方差
//     Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
//     Mat var_I = mean_I2 - mean_I.mul(mean_I);
//     var_I += epsilon;

//     //求a和b
//     Mat a, b;
//     divide(cov_Ip, var_I, a);
//     b = mean_p - a.mul(mean_I);

//     //对包含像素i的所有a、b做平均
//     Mat mean_a, mean_b;
//     boxFilter(a, mean_a, CV_32F, win_size);
//     boxFilter(b, mean_b, CV_32F, win_size);

//     //计算输出 (depth == CV_32F)
//     output = mean_a.mul(guided_32f) + mean_b;
// }


// AVX
void multiplyAVX(const Mat& src1, const Mat& src2, Mat& dst)
{
    CV_Assert(src1.type() == CV_32F && src2.type() == CV_32F && src1.size() == src2.size());
    dst.create(src1.size(), CV_32F);

    int size = src1.rows * src1.cols;
    float* pSrc1 = reinterpret_cast<float*>(src1.data);
    float* pSrc2 = reinterpret_cast<float*>(src2.data);
    float* pDst = reinterpret_cast<float*>(dst.data);

    __m256 m1, m2, m3;
    for (int i = 0; i < size; i += 8) // 8 floats per 256-bit AVX register
    {
        m1 = _mm256_loadu_ps(pSrc1 + i);
        m2 = _mm256_loadu_ps(pSrc2 + i);
        m3 = _mm256_mul_ps(m1, m2);
        _mm256_storeu_ps(pDst + i, m3);
    }
}

void guidedFilter_SIMD(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
    CV_Assert(radius >= 2 && epsilon > 0);
    CV_Assert(source.data != NULL && source.channels() == 1);
    CV_Assert(guided_image.channels() == 1);
    CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

    Mat guided;
    if (guided_image.data == source.data)
    {
        //make a copy
        guided_image.copyTo(guided);
    }
    else
    {
        guided = guided_image;
    }

    // 将输入扩展为32位浮点型，以便以后做乘法
    Mat source_32f, guided_32f;
    makeDepth32f(source, source_32f);
    makeDepth32f(guided, guided_32f);

    // 计算I*p和I*I
    Mat mat_Ip, mat_I2;
    multiplyAVX(guided_32f, source_32f, mat_Ip);
    multiplyAVX(guided_32f, guided_32f, mat_I2);

    // 计算各种均值
    Mat mean_p, mean_I, mean_Ip, mean_I2;
    Size win_size(2 * radius + 1, 2 * radius + 1);
    boxFilter(source_32f, mean_p, CV_32F, win_size);
    boxFilter(guided_32f, mean_I, CV_32F, win_size);
    boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
    boxFilter(mat_I2, mean_I2, CV_32F, win_size);

    // 计算Ip的协方差和I的方差
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    var_I += epsilon;

    // 求a和b
    Mat a, b;
    divide(cov_Ip, var_I, a);
    b = mean_p - a.mul(mean_I);

    // 对包含像素i的所有a、b做平均
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_32F, win_size);
    boxFilter(b, mean_b, CV_32F, win_size);

    // 计算输出 (depth == CV_32F)
    output = mean_a.mul(guided_32f) + mean_b;
}



// Mat getTransmission_dark_SIMD(Mat& srcimg, Mat& darkimg, int *array, int windowsize)
// {
//     //t = 1 - omega * minfilt(I / A);
//     float avg_A;
//     //convertImage是一个CV_32FC3的图
//     if (srcimg.type() % 8 == 0) {
//         avg_A = (array[0] + array[1] + array[2]) / 3.0;
//     }
//     else {
//         avg_A = (array[0] + array[1] + array[2]) / (3.0 * 255.0);
//     }
//     float w = 0.95;
//     int radius = windowsize / 2;
//     int nr = srcimg.rows, nl = srcimg.cols;
//     Mat transmission(nr, nl, CV_32FC1);

//     for (int k = 0; k<nr; k++) 
//     {
//         const float* inData = darkimg.ptr<float>(k);
//         float* outData = transmission.ptr<float>(k);
//         float pix[3] = { 0 };
//         for (int l = 0; l < nl; l++)
//         {
//             *outData++ = 1.0 - w * *inData++;
//         }
//     }
//     // imshow("t", transmission);

//     Mat trans(nr, nl, CV_32FC1);
//     Mat graymat(nr, nl, CV_8UC1);
//     Mat graymat_32F(nr, nl, CV_32FC1);

//     if (srcimg.type() % 8 != 0) 
//     {
//         cvtColor(srcimg, graymat_32F, cv::COLOR_BGR2GRAY);
//         guidedFilter_SIMD(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
//     }
//     else 
//     {
//         cvtColor(srcimg, graymat, cv::COLOR_BGR2GRAY);

//         for (int i = 0; i < nr; i++) 
//         {
//             const uchar* inData = graymat.ptr<uchar>(i);
//             float* outData = graymat_32F.ptr<float>(i);
//             for (int j = 0; j < nl; j++)
//                 *outData++ = *inData++ / 255.0;
//         }
//         guidedFilter_SIMD(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
//     }
//     return trans;
// }

// AVX
Mat getTransmission_dark_SIMD(Mat& srcimg, Mat& darkimg, int *array, int windowsize)
{
    // t = 1 - omega * minfilt(I / A);
    float avg_A;
    // convertImage是一个CV_32FC3的图
    if (srcimg.type() % 8 == 0) {
        avg_A = (array[0] + array[1] + array[2]) / 3.0f;
    }
    else {
        avg_A = (array[0] + array[1] + array[2]) / (3.0f * 255.0f);
    }
    float w = 0.95f;
    int radius = windowsize / 2;
    int nr = srcimg.rows, nl = srcimg.cols;
    Mat transmission(nr, nl, CV_32FC1);

    // AVX指令优化传输图计算
    __m256 w_vec = _mm256_set1_ps(w);
    __m256 one_vec = _mm256_set1_ps(1.0f);

    for (int k = 0; k < nr; k++) 
    {
        const float* inData = darkimg.ptr<float>(k);
        float* outData = transmission.ptr<float>(k);

        int l = 0;
        for (; l <= nl - 8; l += 8)
        {
            __m256 dark_vec = _mm256_loadu_ps(inData + l);
            __m256 result = _mm256_sub_ps(one_vec, _mm256_mul_ps(w_vec, dark_vec));
            _mm256_storeu_ps(outData + l, result);
        }

        // 处理剩余元素
        for (; l < nl; l++)
        {
            outData[l] = 1.0f - w * inData[l];
        }
    }

    Mat trans(nr, nl, CV_32FC1);
    Mat graymat(nr, nl, CV_8UC1);
    Mat graymat_32F(nr, nl, CV_32FC1);

    if (srcimg.type() % 8 != 0) 
    {
        cvtColor(srcimg, graymat_32F, cv::COLOR_BGR2GRAY);
        guidedFilter_SIMD(transmission, graymat_32F, trans, 6 * windowsize, 0.001f);
    }
    else 
    {
        cvtColor(srcimg, graymat, cv::COLOR_BGR2GRAY);

        // AVX指令优化灰度图转换
        __m256 divisor = _mm256_set1_ps(255.0f);

        for (int i = 0; i < nr; i++) 
        {
            const uchar* inData = graymat.ptr<uchar>(i);
            float* outData = graymat_32F.ptr<float>(i);

            int j = 0;
            for (; j <= nl - 8; j += 8)
            {
                __m128i gray_val = _mm_loadl_epi64((__m128i const *)(inData + j));
                __m256i gray_val_32 = _mm256_cvtepu8_epi32(gray_val);
                __m256 gray_val_f = _mm256_cvtepi32_ps(gray_val_32);
                __m256 result = _mm256_div_ps(gray_val_f, divisor);
                _mm256_storeu_ps(outData + j, result);
            }

            // 处理剩余元素
            for (; j < nl; j++)
            {
                outData[j] = inData[j] / 255.0f;
            }
        }
        guidedFilter_SIMD(transmission, graymat_32F, trans, 6 * windowsize, 0.001f);
    }
    return trans;
}

