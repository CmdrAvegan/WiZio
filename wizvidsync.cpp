#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <opencv2/opencv.hpp>
#include <boost/asio.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QSlider>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QComboBox>
#include <unordered_map>
#include <fstream>
#include "nlohmann/json.hpp"
#include <chrono>
#include <opencv2/imgproc.hpp>




using boost::asio::ip::udp;


// Definition of the Light structure
struct Light {
    QString name;
    QString ipAddress;  // Example detail, you can add more as needed
};
struct Vec3bComparator {
    bool operator()(const cv::Vec3b& lhs, const cv::Vec3b& rhs) const {
        return std::tie(lhs[0], lhs[1], lhs[2]) < std::tie(rhs[0], rhs[1], rhs[2]);
    }
};

// Global variables
std::atomic<bool> capturing(false);  // Used to control screen capture
QLabel* colorDisplay;                // Global pointer to the color display label
QLabel* leftColorDisplay;            // Global pointer to the left color display label
QLabel* rightColorDisplay;           // Global pointer to the right color display label
QLabel* colorLabel;                  // Global pointer to the color label text
QLabel* lightsLabel;                 // Global pointer to the Light's label text
QListWidget* lightList;              // List widget to display lights
QCheckBox* brightnessCheckbox;
QSlider* darknessSlider;
QCheckBox* colorBoostCheckbox;
QSlider* colorBoostSlider;
QSlider* darknessThresholdSlider;     // global variable for the color to darkness slider
QCheckBox* dynamicBrightnessCheckbox; // global variable for the dynamic brightness checkbox
QComboBox* leftLightComboBox;         // Global variable for left light
QComboBox* rightLightComboBox;        // Global variable for right light
QCheckBox* ambilightModeCheckbox;     // Global variable for Ambilight mode
QLabel* leftColorLabel;
QLabel* rightColorLabel;
QSlider* mainImageScaleSlider = nullptr;
QSlider* motionImageScaleSlider = nullptr;
QSlider* patternImageScaleSlider = nullptr;
boost::asio::io_context ioContext;
udp::socket* udpSocket;
std::atomic<bool> adjustBrightness(false);  // Flag to enable/disable brightness adjustment
std::atomic<int> darknessThreshold(30);     // Threshold for darkness intensity adjustment
std::atomic<bool> colorBoostEnabled(false);  // Flag to enable/disable color boost
std::atomic<int> colorBoostIntensity(20);   // Intensity of the color boost
std::atomic<int> maxBrightness(100);        // Maximum brightness flag
RECT selectedRegion;
bool selectingRegion = false;
bool regionSelected = false;
std::atomic<bool> previewsOpen(false);
std::atomic<bool> ambilightModeEnabled(false);
std::vector<Light> lights;  // Vector to store the lights
// Global variables for scaling factors
std::atomic<double> mainImageScale(0.5);  // Default scaling factor for main image
std::atomic<double> motionImageScale(0.5); // Default scaling for motion images
std::atomic<double> patternImageScale(0.5); // Default scaling for pattern images
std::thread leftThread;
std::thread rightThread;
std::atomic<bool> leftThreadRunning(false);
std::atomic<bool> rightThreadRunning(false);

cv::Mat previousFrame;  // To store the previous frame

double AdjustBrightnessWithMotion(double brightness, double motion) {
    double motionScale = std::min(motion / 20.0, 1.0);  // Normalize motion intensity
    return std::clamp(brightness + motionScale * 50, 1.0, 100.0);  // Adjust brightness
}


cv::Mat calculateOpticalFlow(const cv::Mat& prev, const cv::Mat& next) {
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prev, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    return flow;
}

std::vector<cv::Point2f> calculateOpticalFlowLK(const cv::Mat& prev, const cv::Mat& next) {
    std::vector<cv::Point2f> prevPoints, nextPoints;
    cv::goodFeaturesToTrack(prev, prevPoints, 100, 0.01, 10);
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev, next, prevPoints, nextPoints, status, err);
    return nextPoints;
}


cv::Mat calculateLBP(const cv::Mat& src) {
    cv::Mat lbpImage;
    src.copyTo(lbpImage);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            uchar center = src.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (src.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (src.at<uchar>(i - 1, j) > center) << 6;
            code |= (src.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (src.at<uchar>(i, j + 1) > center) << 4;
            code |= (src.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (src.at<uchar>(i + 1, j) > center) << 2;
            code |= (src.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (src.at<uchar>(i, j - 1) > center) << 0;
            lbpImage.at<uchar>(i, j) = code;
        }
    }
    return lbpImage;
}

std::vector<cv::Mat> calculateMultiScaleLBP(const cv::Mat& src, const std::vector<int>& radii) {
    std::vector<cv::Mat> lbpImages;
    for (int radius : radii) {
        cv::Mat lbpImage;
        src.copyTo(lbpImage);
        for (int i = radius; i < src.rows - radius; i++) {
            for (int j = radius; j < src.cols - radius; j++) {
                uchar center = src.at<uchar>(i, j);
                unsigned char code = 0;
                code |= (src.at<uchar>(i - radius, j - radius) > center) << 7;
                code |= (src.at<uchar>(i - radius, j) > center) << 6;
                code |= (src.at<uchar>(i - radius, j + radius) > center) << 5;
                code |= (src.at<uchar>(i, j + radius) > center) << 4;
                code |= (src.at<uchar>(i + radius, j + radius) > center) << 3;
                code |= (src.at<uchar>(i + radius, j) > center) << 2;
                code |= (src.at<uchar>(i + radius, j - radius) > center) << 1;
                code |= (src.at<uchar>(i, j - radius) > center) << 0;
                lbpImage.at<uchar>(i, j) = code;
            }
        }
        lbpImages.push_back(lbpImage);
    }
    return lbpImages;
}


// Function to capture the screen using GDI
HBITMAP CaptureScreen() {
    HDC screenDC = GetDC(nullptr);
    HDC memDC = CreateCompatibleDC(screenDC);

    RECT captureRect;
    if (regionSelected) {
        captureRect = selectedRegion;
    } else {
        GetClientRect(GetDesktopWindow(), &captureRect);
    }

    int width = captureRect.right - captureRect.left;
    int height = captureRect.bottom - captureRect.top;

    HBITMAP hBitmap = CreateCompatibleBitmap(screenDC, width, height);
    SelectObject(memDC, hBitmap);
    BitBlt(memDC, 0, 0, width, height, screenDC, captureRect.left, captureRect.top, SRCCOPY);

    ReleaseDC(nullptr, screenDC);
    DeleteDC(memDC);
    return hBitmap;
}


// Function to convert HBITMAP to cv::Mat with scaling down
cv::Mat HBITMAPToMat(HBITMAP hBitmap) {
    BITMAP bmp;
    GetObject(hBitmap, sizeof(BITMAP), &bmp);

    BITMAPINFOHEADER bih;
    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biWidth = bmp.bmWidth;
    bih.biHeight = -bmp.bmHeight;  // Negative height for top-down DIB
    bih.biPlanes = 1;
    bih.biBitCount = 32;
    bih.biCompression = BI_RGB;
    bih.biSizeImage = 0;
    bih.biXPelsPerMeter = 0;
    bih.biYPelsPerMeter = 0;
    bih.biClrUsed = 0;
    bih.biClrImportant = 0;

    cv::Mat mat(bmp.bmHeight, bmp.bmWidth, CV_8UC4);  // Create a Mat with 4 channels (BGRA)
    HDC hdc = CreateCompatibleDC(NULL);
    GetDIBits(hdc, hBitmap, 0, bmp.bmHeight, mat.data, (BITMAPINFO*)&bih, DIB_RGB_COLORS);
    DeleteDC(hdc);

    cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);  // Convert BGRA to BGR
    cv::resize(mat, mat, cv::Size(), mainImageScale.load(), mainImageScale.load()); // Scale down the image based on user preference

    return mat.clone();  // Clone the image to ensure the data is independent
}

cv::Scalar boostSaturation(cv::Scalar color, int boostIntensity) {
    cv::Mat bgr(1, 1, CV_8UC3, cv::Vec3b(color[0], color[1], color[2])); // Create a single pixel image in BGR format
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);  // Convert to HSV

    cv::Vec3b& pixel = hsv.at<cv::Vec3b>(0, 0);
    pixel[1] = std::min(255, pixel[1] + boostIntensity);  // Boost the saturation

    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);  // Convert back to BGR
    cv::Vec3b boostedColor = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(boostedColor[0], boostedColor[1], boostedColor[2]);  // Return the boosted color
}


// Function to calculate the dominant color by finding the most frequent color
cv::Scalar getDominantColor(const cv::Mat& image) {
    
    if (image.empty()) {
        std::cerr << "Error: Image is empty" << std::endl;
        return cv::Scalar(0, 0, 0); // Return black if error
    }

    struct Vec3bHash {
        std::size_t operator()(const cv::Vec3b& color) const {
            return std::hash<int>()((color[0] << 16) | (color[1] << 8) | color[2]);
        }
    };

    std::unordered_map<cv::Vec3b, int, Vec3bHash> colorCount;  // Map to count the frequency of each color

    // Iterate through each pixel in the image
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            colorCount[color]++;
        }
    }

    // Find the color with the highest frequency
    cv::Vec3b dominantColor;
    int maxCount = 0;
    for (const auto& pair : colorCount) {
        if (pair.second > maxCount) {
            dominantColor = pair.first;
            maxCount = pair.second;
        }
    }

    return cv::Scalar(dominantColor[0], dominantColor[1], dominantColor[2]); // BGR to Scalar
}

cv::Scalar getAverageColorBrightness(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Error: Image is empty" << std::endl;
        return cv::Scalar(0, 0, 0); // Return black if error
    }

    cv::Scalar avgBrightness = cv::mean(image); // Calculate the average color brightness

    return avgBrightness; // Return the average brightness
}

void InitUDPSocket() {
    udpSocket = new udp::socket(ioContext, udp::endpoint(udp::v4(), 0));
}

void CleanupUDPSocket() {
    udpSocket->close();
    delete udpSocket;
}

std::atomic<bool> lightIsOff(false);  // Track the state of the light

void SendColorToWiZ(cv::Scalar color, double averageBrightness, const QString& lightIp, const cv::Mat& lbpImage, const cv::Mat& flowMagnitude) {
    if (colorBoostEnabled) {
        color = boostSaturation(color, colorBoostIntensity);
    }

    int red = static_cast<int>(color[2]);
    int green = static_cast<int>(color[1]);
    int blue = static_cast<int>(color[0]);

    int brightness;

    // Directly use max brightness if adjust brightness is off
    if (!adjustBrightness) {
        brightness = maxBrightness.load();
    } else if (dynamicBrightnessCheckbox->isChecked()) {
        // Base brightness adjustment based on average brightness
        brightness = std::clamp(static_cast<int>((averageBrightness / 255.0) * 100), 1, 100);

        // Measure of texture complexity and average flow magnitude
        double lbpComplexity = cv::mean(lbpImage)[0];
        double flowIntensity = cv::mean(flowMagnitude)[0];

        // Adjust further based on texture and motion
        if (lbpComplexity > 50 || flowIntensity > 20) {  // Example thresholds
            brightness = std::min(brightness + 20, 100);
        } else {
            brightness = std::max(brightness - 20, 1);
        }

        // Apply additional darkness threshold adjustment
        int darknessThresholdValue = darknessThresholdSlider->value();
        if (brightness < darknessThresholdValue) {
            int adjustment = darknessSlider->value();
            brightness = std::max(1, brightness - adjustment);
        }
    } else {
        // Static brightness adjustment using darkness threshold
        int maxColorValue = std::max({red, green, blue});
        int darknessThresholdValue = darknessThresholdSlider->value();
        if (maxColorValue < darknessThresholdValue) {
            int colorBrightness = static_cast<int>(0.299 * red + 0.587 * green + 0.114 * blue);
            brightness = std::max(1, 100 - (darknessThreshold.load() * (255 - colorBrightness) / 255));
        } else {
            brightness = maxBrightness.load();
        }
    }

    // Disable calculations of adjusted brightness and average brightness if their options are not enabled
    if (!adjustBrightness) {
        averageBrightness = 0; // Reset averageBrightness to 0 as it should not be calculated
    }

    // Skip calculations for adjustedBrightness if adjustBrightness is false
    double adjustedBrightness = (adjustBrightness) ? (brightness / 100.0) * 255 : (maxBrightness.load() / 100.0) * 255;

    if (red == 0 && green == 0 && blue == 0) {
        red = green = blue = 1;  // Avoid "off" state being sent as (0,0,0)
        adjustedBrightness = 0;
        lightIsOff = true;
    } else {
        lightIsOff = false;
    }

    // Debugging output
    if (adjustBrightness) {
        std::cout << "Adjusted brightness sent to WiZ light: " << adjustedBrightness << std::endl;
        std::cout << "Average Brightness: " << averageBrightness << std::endl;
    }
    std::cout << "Using max brightness: " << brightness << std::endl;

    QString json = QString("{\"method\":\"setPilot\",\"params\":{\"r\":%1,\"g\":%2,\"b\":%3,\"dimming\":%4}}")
        .arg(red).arg(green).arg(blue).arg(adjustedBrightness);

    udp::endpoint endpoint(boost::asio::ip::make_address(lightIp.toStdString()), 38899);
    udpSocket->send_to(boost::asio::buffer(json.toStdString()), endpoint);
}


// Function to update the color preview box
void UpdateColorDisplay(cv::Scalar color) {
    int red = std::clamp(static_cast<int>(color[2]), 0, 255);   // Ensure values are within [0, 255]
    int green = std::clamp(static_cast<int>(color[1]), 0, 255);
    int blue = std::clamp(static_cast<int>(color[0]), 0, 255);
    QString style = QString("background-color: rgb(%1, %2, %3);").arg(red).arg(green).arg(blue);

    QMetaObject::invokeMethod(colorDisplay, [style]() {
        colorDisplay->setStyleSheet(style);
    }, Qt::QueuedConnection);
}

// Function to update the left color preview box
void UpdateLeftColorDisplay(cv::Scalar color) {
    int red = std::clamp(static_cast<int>(color[2]), 0, 255);
    int green = std::clamp(static_cast<int>(color[1]), 0, 255);
    int blue = std::clamp(static_cast<int>(color[0]), 0, 255);
    QString style = QString("background-color: rgb(%1, %2, %3);").arg(red).arg(green).arg(blue);

    QMetaObject::invokeMethod(leftColorDisplay, [style]() {
        leftColorDisplay->setStyleSheet(style);
    }, Qt::QueuedConnection);
}

// Function to update the right color preview box
void UpdateRightColorDisplay(cv::Scalar color) {
    int red = std::clamp(static_cast<int>(color[2]), 0, 255);
    int green = std::clamp(static_cast<int>(color[1]), 0, 255);
    int blue = std::clamp(static_cast<int>(color[0]), 0, 255);
    QString style = QString("background-color: rgb(%1, %2, %3);").arg(red).arg(green).arg(blue);

    QMetaObject::invokeMethod(rightColorDisplay, [style]() {
        rightColorDisplay->setStyleSheet(style);
    }, Qt::QueuedConnection);
}


void ProcessLeftHalf(const cv::Mat& leftHalfColor, const cv::Mat& leftHalfLBP, double& leftBrightness, cv::Scalar& leftColor) {
    leftColor = getDominantColor(leftHalfColor);
    std::cout << "Left Thread: adjustBrightness = " << adjustBrightness << ", maxBrightness = " << maxBrightness.load() << std::endl;

    // Only calculate average brightness if adjustBrightness is true
    if (adjustBrightness) {
        cv::Scalar avgBrightness = getAverageColorBrightness(leftHalfLBP);
        leftBrightness = (avgBrightness[0] + avgBrightness[1] + avgBrightness[2]) / 3.0;
    } else {
        leftBrightness = maxBrightness.load();
    }
}

void ProcessRightHalf(const cv::Mat& rightHalfColor, const cv::Mat& rightHalfLBP, double& rightBrightness, cv::Scalar& rightColor) {
    rightColor = getDominantColor(rightHalfColor);
    std::cout << "Right Thread: adjustBrightness = " << adjustBrightness << ", maxBrightness = " << maxBrightness.load() << std::endl;

    // Only calculate average brightness if adjustBrightness is true
    if (adjustBrightness) {
        cv::Scalar avgBrightness = getAverageColorBrightness(rightHalfLBP);
        rightBrightness = (avgBrightness[0] + avgBrightness[1] + avgBrightness[2]) / 3.0;
    } else {
        rightBrightness = maxBrightness.load();
    }
}


void ShowColorDisplay(QLabel* label, QLabel* textLabel) {
    label->show();
    textLabel->show();
}

void HideColorDisplay(QLabel* label, QLabel* textLabel) {
    label->hide();
    textLabel->hide();
}


// Start the screen capture in a separate thread
void StartScreenCapture() {
    if (!capturing) {
        capturing = true;

        // Show or hide color displays based on the user's light configuration
        if (leftLightComboBox->currentIndex() > 0) {
            ShowColorDisplay(leftColorDisplay, leftColorLabel);
        } else {
            HideColorDisplay(leftColorDisplay, leftColorLabel);
        }

        if (rightLightComboBox->currentIndex() > 0) {
            ShowColorDisplay(rightColorDisplay, rightColorLabel);
        } else {
            HideColorDisplay(rightColorDisplay, rightColorLabel);
        }

        if (leftLightComboBox->currentIndex() == 0 && rightLightComboBox->currentIndex() == 0) {
            ShowColorDisplay(colorDisplay, colorLabel);
        } else {
            HideColorDisplay(colorDisplay, colorLabel);
        }

        // Launch persistent left thread only if left light is defined
        if (leftLightComboBox->currentIndex() > 0 && !leftThreadRunning) {
            leftThreadRunning = true;
            leftThread = std::thread([]() {
                while (leftThreadRunning && capturing) {
                    // Capture the screen
                    HBITMAP hBitmap = CaptureScreen();
                    cv::Mat img = HBITMAPToMat(hBitmap);
                    cv::Mat gray;
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                    // Split the image into left half
                    cv::Mat leftHalfColor = img(cv::Rect(0, 0, img.cols / 2, img.rows));
                    cv::Scalar leftColor = getDominantColor(leftHalfColor);

                    double leftAverageBrightness = 0;
                    if (adjustBrightness) {
                        leftAverageBrightness = (cv::mean(leftHalfColor)[0] +
                                                 cv::mean(leftHalfColor)[1] +
                                                 cv::mean(leftHalfColor)[2]) / 3.0;

                        // Debugging output
                        std::cout << "Left Average Brightness: " << leftAverageBrightness << std::endl;
                    }

                    // Update left color display if the left light is defined
                    UpdateLeftColorDisplay(leftColor);

                    // Send color to left light
                    if (leftLightComboBox->currentIndex() > 0) {
                        QString leftLightIp = lights[leftLightComboBox->currentIndex() - 1].ipAddress;
                        SendColorToWiZ(leftColor, leftAverageBrightness, leftLightIp, cv::Mat(), cv::Mat());
                    }

                    DeleteObject(hBitmap);
                    std::this_thread::sleep_for(std::chrono::milliseconds(0)); // Adjustable delay
                }
            });
        }

        // Launch persistent right thread only if right light is defined
        if (rightLightComboBox->currentIndex() > 0 && !rightThreadRunning) {
            rightThreadRunning = true;
            rightThread = std::thread([]() {
                while (rightThreadRunning && capturing) {
                    // Capture the screen
                    HBITMAP hBitmap = CaptureScreen();
                    cv::Mat img = HBITMAPToMat(hBitmap);
                    cv::Mat gray;
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                    // Split the image into right half
                    cv::Mat rightHalfColor = img(cv::Rect(img.cols / 2, 0, img.cols / 2, img.rows));
                    cv::Scalar rightColor = getDominantColor(rightHalfColor);

                    double rightAverageBrightness = 0;
                    if (adjustBrightness) {
                        rightAverageBrightness = (cv::mean(rightHalfColor)[0] +
                                                  cv::mean(rightHalfColor)[1] +
                                                  cv::mean(rightHalfColor)[2]) / 3.0;

                        // Debugging output
                        std::cout << "Right Average Brightness: " << rightAverageBrightness << std::endl;
                    }

                    // Update right color display if the right light is defined
                    UpdateRightColorDisplay(rightColor);

                    // Send color to right light
                    if (rightLightComboBox->currentIndex() > 0) {
                        QString rightLightIp = lights[rightLightComboBox->currentIndex() - 1].ipAddress;
                        SendColorToWiZ(rightColor, rightAverageBrightness, rightLightIp, cv::Mat(), cv::Mat());
                    }

                    DeleteObject(hBitmap);
                    std::this_thread::sleep_for(std::chrono::milliseconds(0)); // Adjustable delay
                }
            });
        }

        // If no left or right light is defined, use dominant color for all lights
        if (leftLightComboBox->currentIndex() == 0 && rightLightComboBox->currentIndex() == 0) {
            std::thread([]() {
                while (capturing) {
                    // Capture the screen
                    HBITMAP hBitmap = CaptureScreen();
                    cv::Mat img = HBITMAPToMat(hBitmap);
                    cv::Scalar dominantColor = getDominantColor(img);

                    double avgBrightness = 0;
                    if (adjustBrightness) {
                        avgBrightness = (cv::mean(img)[0] +
                                         cv::mean(img)[1] +
                                         cv::mean(img)[2]) / 3.0;

                        // Debugging output
                        std::cout << "Average Brightness: " << avgBrightness << std::endl;
                    }

                    // Send dominant color to all lights
                    for (const Light& light : lights) {
                        SendColorToWiZ(dominantColor, avgBrightness, light.ipAddress, cv::Mat(), cv::Mat());
                    }

                    // Update main dominant color display
                    UpdateColorDisplay(dominantColor);

                    DeleteObject(hBitmap);
                    std::this_thread::sleep_for(std::chrono::milliseconds(0)); // Adjustable delay
                }
            }).detach();
        }
    }
}


// Stop the screen capture
void StopScreenCapture() {
    capturing = false;
    leftThreadRunning = false;
    rightThreadRunning = false;

    if (leftThread.joinable()) {
        leftThread.join();
    }
    if (rightThread.joinable()) {
        rightThread.join();
    }
}


// Win32 Window Procedure
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_COMMAND:
        if (LOWORD(wParam) == 1) {  // Start Button
            capturing = true;
            std::thread(StartScreenCapture).detach();  // Start capturing in a separate thread
        }
        if (LOWORD(wParam) == 2) {  // Stop Button
            capturing = false;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Function to add a new light
void AddLight() {
    bool ok;
    QString name = QInputDialog::getText(nullptr, "Add Light", "Light Name:", QLineEdit::Normal, "", &ok);
    if (ok && !name.isEmpty()) {
        QString ip = QInputDialog::getText(nullptr, "Add Light", "IP Address:", QLineEdit::Normal, "", &ok);
        if (ok && !ip.isEmpty()) {
            lights.push_back({name, ip});
            lightList->addItem(name);

            // Update combo boxes
            leftLightComboBox->addItem(name);
            rightLightComboBox->addItem(name);
        }
    }
}

// Function to remove the selected light
void RemoveLight() {
    QListWidgetItem* item = lightList->currentItem();
    if (item) {
        int row = lightList->row(item);
        lights.erase(lights.begin() + row);
        delete lightList->takeItem(row);

        // Update combo boxes
        leftLightComboBox->removeItem(row + 1);  // +1 to account for "None" option
        rightLightComboBox->removeItem(row + 1);
    }
}


void SelectScreenRegion() {
    HINSTANCE hInstance = GetModuleHandle(nullptr);

    WNDCLASS wndClass = {};
    wndClass.lpfnWndProc = [](HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) -> LRESULT {
        static POINT startPoint, endPoint;
        static HBRUSH hBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
        static HDC hdcMem = nullptr;
        static HBITMAP hbmMem = nullptr;
        static HGDIOBJ hOld = nullptr;

        switch (msg) {
        case WM_CREATE:
            {
                RECT rect;
                GetClientRect(hwnd, &rect);
                HDC hdc = GetDC(hwnd);
                hdcMem = CreateCompatibleDC(hdc);
                hbmMem = CreateCompatibleBitmap(hdc, rect.right, rect.bottom);
                hOld = SelectObject(hdcMem, hbmMem);
                ReleaseDC(hwnd, hdc);
            }
            return 0;

        case WM_LBUTTONDOWN:
            startPoint.x = LOWORD(lParam);
            startPoint.y = HIWORD(lParam);
            endPoint = startPoint;
            selectingRegion = true;
            regionSelected = false;
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;

        case WM_MOUSEMOVE:
            if (selectingRegion) {
                endPoint.x = LOWORD(lParam);
                endPoint.y = HIWORD(lParam);
                InvalidateRect(hwnd, nullptr, TRUE); // Invalidate entire window to clear previous drawings
            }
            return 0;

        case WM_LBUTTONUP:
            if (selectingRegion) {
                endPoint.x = LOWORD(lParam);
                endPoint.y = HIWORD(lParam);
                selectingRegion = false;
                regionSelected = true;

                selectedRegion.left = std::min(startPoint.x, endPoint.x);
                selectedRegion.top = std::min(startPoint.y, endPoint.y);
                selectedRegion.right = std::max(startPoint.x, endPoint.x);
                selectedRegion.bottom = std::max(startPoint.y, endPoint.y);

                DestroyWindow(hwnd);
            }
            return 0;

        case WM_PAINT:
            {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);

                RECT rect;
                GetClientRect(hwnd, &rect);
                FillRect(hdcMem, &rect, (HBRUSH)GetStockObject(BLACK_BRUSH));

                if (selectingRegion) {
                    RECT selectionRect = {
                        std::min(startPoint.x, endPoint.x),
                        std::min(startPoint.y, endPoint.y),
                        std::max(startPoint.x, endPoint.x),
                        std::max(startPoint.y, endPoint.y)
                    };
                    FrameRect(hdcMem, &selectionRect, hBrush);
                }

                BitBlt(hdc, 0, 0, rect.right, rect.bottom, hdcMem, 0, 0, SRCCOPY);
                EndPaint(hwnd, &ps);
            }
            return 0;

        case WM_DESTROY:
            if (hdcMem) {
                SelectObject(hdcMem, hOld);
                DeleteObject(hbmMem);
                DeleteDC(hdcMem);
            }
            PostQuitMessage(0);
            return 0;

        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
        }
    };

    wndClass.hInstance = hInstance;
    wndClass.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
    wndClass.lpszClassName = "RegionSelector";
    wndClass.style = CS_HREDRAW | CS_VREDRAW; // Add these styles for redraw on size change
    RegisterClass(&wndClass);

    HWND hwnd = CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_LAYERED,
        "RegionSelector",
        "Select Screen Region",
        WS_POPUP | WS_VISIBLE,
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, hInstance, nullptr
    );

    // Make the window partially transparent but keep it interactive
    SetLayeredWindowAttributes(hwnd, 0, 128, LWA_ALPHA); // Set alpha to 128 for partial transparency

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}


const int DEFAULT_DARKNESS_THRESHOLD = 30;
const int DEFAULT_COLOR_BOOST_INTENSITY = 20;
const bool DEFAULT_ADJUST_BRIGHTNESS = false;
const bool DEFAULT_COLOR_BOOST_ENABLED = false;

void ResetToDefaults() {
    darknessThreshold = DEFAULT_DARKNESS_THRESHOLD;
    colorBoostIntensity = DEFAULT_COLOR_BOOST_INTENSITY;
    adjustBrightness = DEFAULT_ADJUST_BRIGHTNESS;
    colorBoostEnabled = DEFAULT_COLOR_BOOST_ENABLED;
    lights.clear();

    // Update the UI elements to reflect these defaults
    brightnessCheckbox->setChecked(DEFAULT_ADJUST_BRIGHTNESS);
    darknessSlider->setValue(DEFAULT_DARKNESS_THRESHOLD);
    colorBoostCheckbox->setChecked(DEFAULT_COLOR_BOOST_ENABLED);
    colorBoostSlider->setValue(DEFAULT_COLOR_BOOST_INTENSITY);
    darknessThresholdSlider->setValue(20); // Reset the darkness threshold slider to default value
    lightList->clear();
}

// Function to save settings to a JSON file
void SaveSettings() {
    nlohmann::json settings;

    // Save darkness threshold and color boost intensity
    settings["darknessThreshold"] = darknessThreshold.load();
    settings["colorBoostIntensity"] = colorBoostIntensity.load();
    settings["darknessThresholdValue"] = darknessThresholdSlider->value(); // Save the darkness threshold slider value
    settings["dynamicBrightnessEnabled"] = dynamicBrightnessCheckbox->isChecked(); // Save dynamic brightness state
    settings["mainImageScale"] = mainImageScale.load();
    settings["motionImageScale"] = motionImageScale.load();
    settings["patternImageScale"] = patternImageScale.load();

    // Save whether darkness adjustment and color boost are enabled
    settings["adjustBrightness"] = adjustBrightness.load();
    settings["colorBoostEnabled"] = colorBoostEnabled.load();
    // Save if ambilight mode is enabled
    settings["ambilightModeEnabled"] = ambilightModeEnabled.load();

    // Save lights
    settings["lights"] = nlohmann::json::array();
    for (const Light& light : lights) {
        settings["lights"].push_back({{"name", light.name.toStdString()}, {"ipAddress", light.ipAddress.toStdString()}});
    }

    // Save the settings to a file
    std::ofstream settingsFile("settings.json");
    settingsFile << settings.dump(4); // Pretty-print with 4 spaces
}


// Function to create a default settings file
void CreateDefaultSettingsFile() {
    nlohmann::json settings;

    // Default settings
    settings["darknessThreshold"] = DEFAULT_DARKNESS_THRESHOLD;
    settings["colorBoostIntensity"] = DEFAULT_COLOR_BOOST_INTENSITY;
    settings["adjustBrightness"] = DEFAULT_ADJUST_BRIGHTNESS;
    settings["colorBoostEnabled"] = DEFAULT_COLOR_BOOST_ENABLED;
    settings["lights"] = nlohmann::json::array();

    // Save the default settings to a file
    std::ofstream settingsFile("settings.json");
    settingsFile << settings.dump(4); // Pretty-print with 4 spaces
}

// Function to load settings from a JSON file
void LoadSettings() {
    std::ifstream settingsFile("settings.json");
    if (settingsFile.is_open()) {
        nlohmann::json settings;
        settingsFile >> settings;

        // Load darkness threshold and color boost intensity
        darknessThreshold = settings.value("darknessThreshold", DEFAULT_DARKNESS_THRESHOLD);
        colorBoostIntensity = settings.value("colorBoostIntensity", DEFAULT_COLOR_BOOST_INTENSITY);
        int darknessThresholdValue = settings.value("darknessThresholdValue", 20); // Load the darkness threshold slider value
        bool dynamicBrightnessEnabled = settings.value("dynamicBrightnessEnabled", false); // Load dynamic brightness state
        mainImageScale = settings.value("mainImageScale", 0.5);  // Default to 0.5
        motionImageScale = settings.value("motionImageScale", 0.5);
        patternImageScale = settings.value("patternImageScale", 0.5);

        // Load whether darkness adjustment and color boost are enabled
        adjustBrightness = settings.value("adjustBrightness", DEFAULT_ADJUST_BRIGHTNESS);
        colorBoostEnabled = settings.value("colorBoostEnabled", DEFAULT_COLOR_BOOST_ENABLED);
        // Load if ambilight setting is enabled
        ambilightModeEnabled = settings.value("ambilightModeEnabled", false);
        if (ambilightModeCheckbox) ambilightModeCheckbox->setChecked(ambilightModeEnabled);

        // Load lights
        lights.clear();
        lightList->clear();
        leftLightComboBox->clear();
        rightLightComboBox->clear();
        leftLightComboBox->addItem("None");
        rightLightComboBox->addItem("None");
        for (const auto& light : settings["lights"]) {
            lights.push_back({QString::fromStdString(light["name"]), QString::fromStdString(light["ipAddress"])});
            lightList->addItem(QString::fromStdString(light["name"]));
            leftLightComboBox->addItem(QString::fromStdString(light["name"]));
            rightLightComboBox->addItem(QString::fromStdString(light["name"]));
        }

        // Update UI elements to reflect loaded settings
        if (brightnessCheckbox) brightnessCheckbox->setChecked(adjustBrightness);
        if (darknessSlider) darknessSlider->setValue(darknessThreshold);
        if (colorBoostCheckbox) colorBoostCheckbox->setChecked(colorBoostEnabled);
        if (colorBoostSlider) colorBoostSlider->setValue(colorBoostIntensity);
        if (darknessThresholdSlider) darknessThresholdSlider->setValue(darknessThresholdValue);
        if (dynamicBrightnessCheckbox) dynamicBrightnessCheckbox->setChecked(dynamicBrightnessEnabled);
        if (mainImageScaleSlider) mainImageScaleSlider->setValue(static_cast<int>(mainImageScale.load() * 100));
        if (motionImageScaleSlider) motionImageScaleSlider->setValue(static_cast<int>(motionImageScale.load() * 100));
        if (patternImageScaleSlider) patternImageScaleSlider->setValue(static_cast<int>(patternImageScale.load() * 100));
        
        std::cout << "Settings loaded successfully." << std::endl;
    } else {
        std::cerr << "Could not open settings file. Creating a default settings file." << std::endl;
        CreateDefaultSettingsFile();
    }
}



std::thread previewThread;

void StartPreview() {
    if (previewsOpen) {
        std::cout << "Preview windows are already open." << std::endl;
        return;
    }

    previewsOpen = true;

    if (previewThread.joinable()) {
        previewThread.join();  // Ensure the previous thread is stopped
    }

    previewThread = std::thread([]() {
        cv::Mat previousFrame;

        // Initialize preview windows
        cv::namedWindow("Optical Flow Magnitude", cv::WINDOW_NORMAL);
        cv::namedWindow("LBP Pattern Analysis", cv::WINDOW_NORMAL);

        while (previewsOpen) {
            // Check if preview windows are still open
            bool magnitudeWindowOpen = cv::getWindowProperty("Optical Flow Magnitude", cv::WND_PROP_VISIBLE) >= 1;
            bool lbpWindowOpen = cv::getWindowProperty("LBP Pattern Analysis", cv::WND_PROP_VISIBLE) >= 1;

            // Exit the loop if both windows are closed
            if (!magnitudeWindowOpen && !lbpWindowOpen) {
                previewsOpen = false;
                break;
            }

            if (capturing) {
                // Capture the screen for preview
                HBITMAP hBitmap = CaptureScreen();
                cv::Mat img = HBITMAPToMat(hBitmap);
                cv::Mat gray;
                cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                if (!previousFrame.empty()) {
                    // Calculate optical flow
                    cv::Mat flow = calculateOpticalFlow(previousFrame, gray);
                    cv::Mat flowChannels[2];
                    cv::split(flow, flowChannels);
                    cv::Mat magnitude, angle;
                    cv::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);
                    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
                    magnitude.convertTo(magnitude, CV_8UC1);

                    // Calculate LBP image
                    auto lbpImages = calculateMultiScaleLBP(gray, {1, 2, 3});
                    cv::Mat combinedLBP;
                    cv::addWeighted(lbpImages[0], 0.33, lbpImages[1], 0.33, 0, combinedLBP);
                    cv::addWeighted(combinedLBP, 1.0, lbpImages[2], 0.33, 0, combinedLBP);

                    // Display previews only if windows are open
                    if (magnitudeWindowOpen && !magnitude.empty()) {
                        cv::imshow("Optical Flow Magnitude", magnitude);
                    }

                    if (lbpWindowOpen && !combinedLBP.empty()) {
                        cv::imshow("LBP Pattern Analysis", combinedLBP);
                    }
                }

                previousFrame = gray.clone();
                DeleteObject(hBitmap);
            }

            // Keep windows responsive, even if not capturing
            cv::waitKey(50); // Delay to allow OpenCV to process GUI events
        }

        // Destroy preview windows once previews are stopped
        cv::destroyAllWindows();
        std::cout << "Preview completed." << std::endl;
    });
}


void SetupUI(QVBoxLayout* layout) {
    // Add start and stop buttons
    QPushButton* startButton = new QPushButton("Start");
    QPushButton* stopButton = new QPushButton("Stop");
    layout->addWidget(startButton);
    layout->addWidget(stopButton);

    // Add region selection button
    QPushButton* selectRegionButton = new QPushButton("Select Region");
    layout->addWidget(selectRegionButton);

    // Add preview button
    QPushButton* previewButton = new QPushButton("Preview");
    layout->addWidget(previewButton);

    // Create a vertical layout for each color display box
    QVBoxLayout* leftColorLayout = new QVBoxLayout;
    QVBoxLayout* dominantColorLayout = new QVBoxLayout;
    QVBoxLayout* rightColorLayout = new QVBoxLayout;

    // Initialize and add color display for left color
    leftColorLabel = new QLabel("Left Color:");
    leftColorDisplay = new QLabel;
    leftColorDisplay->setFixedSize(100, 50);
    leftColorDisplay->setStyleSheet("background-color: rgb(0, 0, 0);");  // Initial color
    leftColorLayout->addWidget(leftColorLabel);
    leftColorLayout->addWidget(leftColorDisplay);

    // Initialize and add color display for dominant color
    colorLabel = new QLabel("Dominant Color:");
    colorDisplay = new QLabel;
    colorDisplay->setFixedSize(100, 50);
    colorDisplay->setStyleSheet("background-color: rgb(0, 0, 0);");  // Initial color
    dominantColorLayout->addWidget(colorLabel);
    dominantColorLayout->addWidget(colorDisplay);

    // Initialize and add color display for right color
    rightColorLabel = new QLabel("Right Color:");
    rightColorDisplay = new QLabel;
    rightColorDisplay->setFixedSize(100, 50);
    rightColorDisplay->setStyleSheet("background-color: rgb(0, 0, 0);");  // Initial color
    rightColorLayout->addWidget(rightColorLabel);
    rightColorLayout->addWidget(rightColorDisplay);

    // Create a horizontal layout to hold the vertical layouts
    QHBoxLayout* colorDisplayLayout = new QHBoxLayout;
    colorDisplayLayout->addLayout(leftColorLayout);
    colorDisplayLayout->addLayout(dominantColorLayout);
    colorDisplayLayout->addLayout(rightColorLayout);

    // Add the horizontal layout to the main layout
    layout->addLayout(colorDisplayLayout);

    // Add list widget for lights
    lightsLabel = new QLabel("WiZ Light List:");
    lightList = new QListWidget;
    layout->addWidget(lightsLabel);
    layout->addWidget(lightList);

    // Add add/remove buttons for lights
    QPushButton* addButton = new QPushButton("Add Light");
    QPushButton* removeButton = new QPushButton("Remove Light");
    layout->addWidget(addButton);
    layout->addWidget(removeButton);

    // Add brightness adjustment controls
    brightnessCheckbox = new QCheckBox("Adjust Brightness Based on Darkness");
    layout->addWidget(brightnessCheckbox);
    darknessSlider = new QSlider(Qt::Horizontal);
    darknessSlider->setRange(0, 100);
    layout->addWidget(new QLabel("Darkness Intensity"));
    layout->addWidget(darknessSlider);

    // Add a new slider for the darkness threshold
    QLabel* darknessThresholdLabel = new QLabel("Darkness Threshold:");
    layout->addWidget(darknessThresholdLabel);
    darknessThresholdSlider = new QSlider(Qt::Horizontal);
    darknessThresholdSlider->setRange(0, 255); // Set the range to 0-255
    darknessThresholdSlider->setValue(20); // Default value
    layout->addWidget(darknessThresholdSlider);

    // Add maximum brightness slider
    QLabel* maxBrightnessLabel = new QLabel("Maximum Brightness:");
    layout->addWidget(maxBrightnessLabel);
    QSlider* maxBrightnessSlider = new QSlider(Qt::Horizontal);
    maxBrightnessSlider->setRange(0, 100); // Set the range to 0-100
    maxBrightnessSlider->setValue(100); // Default value is 100%
    layout->addWidget(maxBrightnessSlider);

    // Add dynamic brightness checkbox
    dynamicBrightnessCheckbox = new QCheckBox("Enable Dynamic Brightness");
    layout->addWidget(dynamicBrightnessCheckbox);

    // Add color boost controls
    colorBoostSlider = new QSlider(Qt::Horizontal);
    colorBoostSlider->setRange(0, 100);
    layout->addWidget(new QLabel("Color Boost Intensity"));
    layout->addWidget(colorBoostSlider);
    colorBoostCheckbox = new QCheckBox("Enable Color Boost");
    layout->addWidget(colorBoostCheckbox);

    // Add combo boxes for selecting lights for left and right halves
    QLabel* leftLightLabel = new QLabel("Select Light for Left Half:");
    layout->addWidget(leftLightLabel);
    leftLightComboBox = new QComboBox();
    leftLightComboBox->addItem("None");
    layout->addWidget(leftLightComboBox);

    QLabel* rightLightLabel = new QLabel("Select Light for Right Half:");
    layout->addWidget(rightLightLabel);
    rightLightComboBox = new QComboBox();
    rightLightComboBox->addItem("None");
    layout->addWidget(rightLightComboBox);

    // Add scaling sliders
    QLabel* mainImageScaleLabel = new QLabel("Main Image Scaling:");
    mainImageScaleSlider = new QSlider(Qt::Horizontal); // Assign to the global pointer
    mainImageScaleSlider->setRange(10, 100);  // Scale from 0.1 to 1.0 (multiplied by 100 for UI)
    mainImageScaleSlider->setValue(static_cast<int>(mainImageScale.load() * 100));
    layout->addWidget(mainImageScaleLabel);
    layout->addWidget(mainImageScaleSlider);

    QLabel* motionImageScaleLabel = new QLabel("Motion Image Scaling:");
    motionImageScaleSlider = new QSlider(Qt::Horizontal); // Assign to the global pointer
    motionImageScaleSlider->setRange(10, 100);
    motionImageScaleSlider->setValue(static_cast<int>(motionImageScale.load() * 100));
    layout->addWidget(motionImageScaleLabel);
    layout->addWidget(motionImageScaleSlider);

    QLabel* patternImageScaleLabel = new QLabel("Pattern Image Scaling:");
    patternImageScaleSlider = new QSlider(Qt::Horizontal); // Assign to the global pointer
    patternImageScaleSlider->setRange(10, 100);
    patternImageScaleSlider->setValue(static_cast<int>(patternImageScale.load() * 100));
    layout->addWidget(patternImageScaleLabel);
    layout->addWidget(patternImageScaleSlider);

    ambilightModeCheckbox = new QCheckBox("Enable Ambilight Mode");
    layout->addWidget(ambilightModeCheckbox);

    // Add reset to defaults button
    QPushButton* resetButton = new QPushButton("Reset to Defaults");
    layout->addWidget(resetButton);

    // Connect signals and slots
    QObject::connect(startButton, &QPushButton::clicked, StartScreenCapture);
    QObject::connect(stopButton, &QPushButton::clicked, StopScreenCapture);
    QObject::connect(previewButton, &QPushButton::clicked, StartPreview); // Corrected signal-slot connection
    QObject::connect(addButton, &QPushButton::clicked, &AddLight);
    QObject::connect(removeButton, &QPushButton::clicked, &RemoveLight);
    QObject::connect(brightnessCheckbox, &QCheckBox::stateChanged, [](int state) {
        std::cout << "Adjust Brightness Checkbox state: " << state << std::endl;
        adjustBrightness = (state == Qt::Checked);
    });
    QObject::connect(darknessSlider, &QSlider::valueChanged, [](int value) {
        darknessThreshold = value;
    });
    QObject::connect(colorBoostCheckbox, &QCheckBox::stateChanged, [](int state) {
        colorBoostEnabled = (state == Qt::Checked);
    });
    QObject::connect(colorBoostSlider, &QSlider::valueChanged, [](int value) {
        colorBoostIntensity = value;
    });
    QObject::connect(darknessThresholdSlider, &QSlider::valueChanged, [](int value) {
        // Use the new threshold value as needed
    });
    QObject::connect(dynamicBrightnessCheckbox, &QCheckBox::stateChanged, [](int state) {
        // Handle dynamic brightness checkbox state change
    });
    QObject::connect(ambilightModeCheckbox, &QCheckBox::stateChanged, [](int state) {
        ambilightModeEnabled = (state == Qt::Checked);
    });
    QObject::connect(mainImageScaleSlider, &QSlider::valueChanged, [](int value) {
        mainImageScale = value / 100.0;
    });
    QObject::connect(motionImageScaleSlider, &QSlider::valueChanged, [](int value) {
        motionImageScale = value / 100.0;
    });
    QObject::connect(patternImageScaleSlider, &QSlider::valueChanged, [](int value) {
        patternImageScale = value / 100.0;
    });
    QObject::connect(maxBrightnessSlider, &QSlider::valueChanged, [](int value) {
    std::cout << "MaxBrightnessSlider value: " << value << std::endl;
    maxBrightness = value;
    });
    QObject::connect(selectRegionButton, &QPushButton::clicked, []() {
        if (previewsOpen) {
            QMessageBox::warning(nullptr, "Error", "Please close the preview windows before selecting a region.");
            return;
        }
        SelectScreenRegion();
    });
    QObject::connect(resetButton, &QPushButton::clicked, []() {
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(nullptr, "Reset to Defaults", "Are you sure you want to reset to default settings?",
                                      QMessageBox::Yes | QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            ResetToDefaults();
        }
    });
}




void Cleanup() {
    if (previewsOpen) {
        previewsOpen = false;
        if (previewThread.joinable()) {
            previewThread.join();
        }
    }
    cv::destroyAllWindows();
}


int main(int argc, char* argv[]) {
    // Initialize UDP socket
    InitUDPSocket();
    std::cout << "WiZ light UDP Socket initialized." << std::endl;

    // Initialize Qt application
    QApplication app(argc, argv);

    // Create main window
    QMainWindow window;
    window.setWindowTitle("WiZ Light Screen Sync");
    window.resize(400, 300);

    // Create central widget and layout
    QWidget* centralWidget = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);
    SetupUI(layout);
    window.setCentralWidget(centralWidget);

    // Show the window
    window.show();
    std::cout << "Main window loaded." << std::endl;

    // Load settings from file after UI is initialized
    LoadSettings();

    // Run the application
    int result = app.exec();

    // Save settings to file
    SaveSettings();
    std::cout << "Settings saved." << std::endl;

    // Cleanup UDP socket
    CleanupUDPSocket();
    std::cout << "WiZ light UDP Socket cleaned up." << std::endl;
    std::cout << "Closing threads and cleaning up, please wait..." << std::endl;
    return result;
}
