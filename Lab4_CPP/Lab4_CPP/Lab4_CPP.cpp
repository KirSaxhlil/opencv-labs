#include <opencv2/opencv.hpp> 
#include <iostream>

#define and &&
#define or ||

using namespace cv;
using namespace std;

float half_grad(float img_part[3][3], float oper[3][3]) {
    float sum = 0;
    for (int y = 0; y < 3; y++) {//for y in range(len(img_part)) :
        for (int x = 0; x < 3; x++) {//for x in range(len(img_part[y])) :
            sum += img_part[y][x] * oper[y][x];
        }
    }
    return sum;
}

int get_dir(float x, float y, float tg) {
    if ((x > 0 and y < 0 and tg < -2.414) or (x < 0 and y < 0 and tg>2.414) or (x == 0 and y < 0)) return 0;
    else if (x > 0 and y < 0 and tg < -0.414) return 1;
    else if ((x > 0 and y<0 and tg>-0.414) or (x > 0 and y > 0 and tg < 0.414) or (x > 0 and y == 0)) return 2;
    else if (x > 0 and y > 0 and tg < 2.414) return 3;
    else if ((x > 0 and y > 0 and tg > 2.414) or (x < 0 and y>0 and tg < -2.414) or (x == 0 and y > 0)) return 4;
    else if (x < 0 and y>0 and tg < -0.414) return 5;
    else if ((x < 0 and y>0 and tg > -0.414) or (x < 0 and y < 0 and tg < 0.414) or (x < 0 and y == 0)) return 6;
    else if (x < 0 and y < 0 and tg < 2.414) return 7;
}

float grad_length(Mat img, int x, int y, float sobelX[3][3], float sobelY[3][3]) {
    //img.at<float>(0,0);
    /*float img_part[3][3] = { {img.at<uchar>(x - 1,y - 1), img.at<uchar>(x,y - 1), img.at<uchar>(x + 1,y - 1)},
                             {img.at<uchar>(x - 1,y), img.at<uchar>(x,y), img.at<uchar>(x + 1,y)},
                             {img.at<uchar>(x - 1,y + 1), img.at<uchar>(x,y + 1), img.at<uchar>(x + 1,y + 1)} };*/
    float img_part[3][3] = { {img.at<uchar>(y - 1,x - 1), img.at<uchar>(y - 1, x), img.at<uchar>(y - 1, x + 1)},
                             {img.at<uchar>(y, x - 1), img.at<uchar>(y, x), img.at<uchar>(y, x + 1)},
                             {img.at<uchar>(y + 1, x - 1), img.at<uchar>(y + 1, x), img.at<uchar>(y + 1, x + 1)} };
    //img_part = img[y - 1:y + 2, x - 1 : x + 2];
    float sXr = half_grad(img_part, sobelX);
    float sYr = half_grad(img_part, sobelY);
    return sqrt(sXr*sXr + sYr*sYr);
}

int main(int argc, char** argv)
{
    // Read the image file 
    //Mat image = imread("C:/Files/University/7 sem/мультипетухон/Lab4/quality.jpg");
    Mat image = imread("C:/Files/University/7 sem/мультипетухон/Lab4/LuminescentCore_Camera_Point_002_s.png");
    // Check for failure 
    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    }
    // Show our image inside a window. 

    int kernel_size = 10;
    float sigma = 1;
    float sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    float sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat blured;
    blur(gray, blured, Size(kernel_size, kernel_size));
    Mat borders;
    blured.copyTo(borders);
    float **grads = new float*[borders.size[0]];
    int ** dirs = new int* [borders.size[0]];
    for (int y = 0; y < borders.size[0]; y++) {
        grads[y] = new float [borders.size[1]];
        dirs[y] = new int[borders.size[1]];
        for (int x = 0; x < borders.size[1]; x++) {
            grads[y][x] = 0;
            dirs[y][x] = -1;
        }
    }

    float max_grad = 0;
    //blured.at<uchar>(1, 1);

    /*Point org(30, 100);
    putText(blured, to_string(borders.cols), org,
        FONT_HERSHEY_SCRIPT_COMPLEX, 2.1,
        Scalar(255, 0, 255), 2, LINE_AA);*/

    for (int y = 0; y < borders.rows; y++) {
        for (int x = 0; x < borders.cols; x++) {
            if (x == 0 or x == borders.cols - 1 or y == 0 or y == borders.rows - 1) {
                grads[y][x] = 0;
                dirs[y][x] = -1;
                continue;
            }

            float img_part[3][3] = { {blured.at<uchar>(y - 1,x - 1), blured.at<uchar>(y - 1, x), blured.at<uchar>(y - 1, x + 1)},
                             {blured.at<uchar>(y, x - 1), blured.at<uchar>(y, x), blured.at<uchar>(y, x + 1)},
                             {blured.at<uchar>(y + 1, x - 1), blured.at<uchar>(y + 1, x), blured.at<uchar>(y + 1, x + 1)} };

            float Gx = half_grad(img_part, sobelX);
            float Gy = half_grad(img_part, sobelY);
            if (Gx == 0 and Gy == 0) {
                grads[y][x] = 0;
                dirs[y][x] = -1;
                continue;
            }
            float tg = Gy / Gx;
            dirs[y][x] = get_dir(Gx, Gy, tg);
            grads[y][x] = sqrt(Gx * Gx + Gy * Gy);
            if (grads[y][x] > max_grad) {
                max_grad = grads[y][x];
            }
        }
    }

    for (int y = 0; y < borders.rows; y++) {
        for (int x = 0; x < borders.cols; x++) {
            if (x == 0 or x == borders.cols - 1 or y == 0 or y == borders.rows - 1) {
                borders.at<uchar>(y, x) = 0;
                continue;
            }

            if (dirs[y][x] == 0 or dirs[y][x] == 4) {
                if (grads[y][x] > grads[y - 1][x] and grads[y][x] > grads[y + 1][x]) {
                    borders.at<uchar>(y, x) = 255;
                }
                else {
                    borders.at<uchar>(y, x) = 0;
                }
            }
            else if (dirs[y][x] == 1 or dirs[y][x] == 5) {
                if (grads[y][x] > grads[y - 1][x + 1] and grads[y][x] > grads[y + 1][x - 1]) {
                    borders.at<uchar>(y, x) = 255;
                }
                else {
                    borders.at<uchar>(y, x) = 0;
                }
            }
            else if (dirs[y][x] == 2 or dirs[y][x] == 6) {
                if (grads[y][x] > grads[y][x + 1] and grads[y][x] > grads[y][x - 1]) {
                    borders.at<uchar>(y, x) = 255;
                }
                else {
                    borders.at<uchar>(y, x) = 0;
                }
            }
            else if (dirs[y][x] == 3 or dirs[y][x] == 7) {
                if (grads[y][x] > grads[y - 1][x - 1] and grads[y][x] > grads[y + 1][x + 1]) {
                    borders.at<uchar>(y, x) = 255;
                }
                else {
                    borders.at<uchar>(y, x) = 0;
                }
            }
            else {
                borders.at<uchar>(y, x) = 0;
            }
        }
    }

    float low_level = max_grad / 15;
    float high_level = max_grad / 10;

    for (int y = 0; y < borders.rows; y++) {
        for (int x = 0; x < borders.cols; x++) {
            if (x == 0 or x == borders.cols - 1 or y == 0 or y == borders.rows - 1) {
                borders.at<uchar>(y, x) = 0;
                continue;
            }
            if (borders.at<uchar>(y, x) == 255) {
                if (grads[y][x] < low_level) {
                    borders.at<uchar>(y, x) = 0;
                }
                else if (grads[y][x] > high_level) {
                    continue;
                }
                else {
                    bool nbr = false;
                    for (int xx = -1; xx <= 1; xx++) {
                        if (nbr == true) {
                            break;
                        }
                        for (int yy = -1; yy <= 1; yy++) {
                            if (borders.at<uchar>(y + yy, x + xx) == 255) {
                                nbr = true;
                                break;
                            }
                        }
                        if (nbr == true) {
                            continue;
                        }
                        else {
                            borders.at<uchar>(y, x) = 0;
                        }
                    }
                }
            }
        }
    }

    imshow("Image Window Name here", borders);

    // Wait for any keystroke in the window 
    waitKey(0);
    return 0;
}