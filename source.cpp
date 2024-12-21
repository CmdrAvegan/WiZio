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
#include <QtWidgets/QScrollArea>
#include <Qtcore/QDir>
#include <Qtcore/QTimer>
#include <unordered_map>
#include <fstream>
#include "nlohmann/json.hpp"
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <mutex> // Include for thread safety
#include <future>  // For std::promise and std::future
#include <vector>
#include <string>
#include <QtCore/QRegularExpression>



using boost::asio::ip::udp;

const QString presetsDirectory = "./presets/";

// Function to remove HTML tags from a QString
QString removeHtmlTags(const QString& input) {
    QString result = input; // Create a mutable copy of the input
    QRegularExpression re("<[^>]*>"); // Regular expression to match HTML tags
    result.replace(re, ""); // Replace HTML tags with an empty string
    return result; // Return the modified string
}

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
struct MonitorInfo {
    std::string name;
    RECT bounds;
};
struct Preset {
    QString name;
    int darknessThreshold;
    int colorBoostIntensity;
    bool adjustBrightness;
    bool colorBoostEnabled;
    bool ambilightModeEnabled;
    bool dynamicBrightnessEnabled;
    double mainImageScale;
    double motionImageScale;
    double patternImageScale;
    int maxBrightness; 
    std::vector<Light> lights;
};



std::mutex lightsMutex; // Global mutex for protecting shared resources

// Global variables
std::atomic<bool> capturing(false);  // Used to control screen capture
std::atomic<bool> pauseCapture(false);  // Tracks if the capture process is paused
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
QComboBox* monitorComboBox;           // Global variable for the monitor combobox
QLabel* leftColorLabel;
QLabel* rightColorLabel;
QSlider* mainImageScaleSlider = nullptr;
QSlider* motionImageScaleSlider = nullptr;
QSlider* patternImageScaleSlider = nullptr;
QSlider* maxBrightnessSlider = nullptr;
QLabel* currentPresetLabel = nullptr;
QPushButton* discoverButton; // Global declaration


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
bool isInitialized = false; // track initialization
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
std::mutex regionMutex; // Mutex to protect the region during screen capture and selection
std::vector<MonitorInfo> monitors; // Global vector to store monitor information
cv::Mat previousFrame;  // To store the previous frame
std::vector<Preset> presets; // Global vector to store presets

std::string GetSceneName(int sceneId) {
    static const std::unordered_map<int, std::string> sceneNames = {
        {1, "Ocean"},
        {2, "Romance"},
        {3, "Sunset"},
        {4, "Party"},
        {5, "Fireplace"},
        {6, "Cozy"},
        {7, "Forest"},
        {8, "Pastel colors"},
        {9, "Wake-up"},
        {10, "Bedtime"},
        {11, "Warm white"},
        {12, "Daylight"},
        {13, "Cool white"},
        {14, "Night light"},
        {15, "Focus"},
        {16, "Relax"},
        {17, "True colors"},
        {18, "TV time"},
        {19, "Plant growth"},
        {20, "Spring"},
        {21, "Summer"},
        {22, "Fall"},
        {23, "Deep dive"},
        {24, "Jungle"},
        {25, "Mojito"},
        {26, "Club"},
        {27, "Christmas"},
        {28, "Halloween"},
        {29, "Candlelight"},
        {30, "Golden white"},
        {31, "Pulse"},
        {32, "Steampunk"},
        {33, "Diwali"},
        {34, "White"},
        {35, "Alarm"},
        {1000, "Rhythm"}
    };

    auto it = sceneNames.find(sceneId);
    return it != sceneNames.end() ? it->second : "Scene " + std::to_string(sceneId);
}


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

// Callback function to enumerate monitors
BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    MONITORINFOEX monitorInfo;
    monitorInfo.cbSize = sizeof(MONITORINFOEX);

    if (GetMonitorInfo(hMonitor, &monitorInfo)) {
        MonitorInfo info;
        info.name = monitorInfo.szDevice;
        info.bounds = monitorInfo.rcMonitor;
        monitors.push_back(info);

        // Debug output
        std::cout << "Monitor found: " << monitorInfo.szDevice 
                  << " Bounds: (" << info.bounds.left << ", " << info.bounds.top 
                  << ", " << info.bounds.right << ", " << info.bounds.bottom << ")" << std::endl;
    } else {
        std::cerr << "Failed to retrieve monitor info!" << std::endl;
    }
    return TRUE;
}



// Function to enumerate all monitors
void EnumerateMonitors() {
    monitors.clear();
    EnumDisplayMonitors(nullptr, nullptr, MonitorEnumProc, 0);
}

void PopulateMonitorComboBox() {
    monitorComboBox->clear();
    EnumerateMonitors();  // Call this to populate the monitors vector

    for (size_t i = 0; i < monitors.size(); ++i) {
        std::string monitorName = "Monitor " + std::to_string(i + 1);  // Create a user-friendly name
        monitorComboBox->addItem(QString::fromStdString(monitorName));
    }

    // Debug output
    std::cout << "Monitors enumerated: " << monitors.size() << std::endl;
}





// Function to capture the screen using GDI
HBITMAP CaptureScreen() {
    std::lock_guard<std::mutex> lock(regionMutex);

    HDC screenDC = GetDC(nullptr);
    HDC memDC = CreateCompatibleDC(screenDC);

    RECT captureRect = selectedRegion;
    if (!regionSelected) {
        // Default to the primary monitor if no region is selected
        GetClientRect(GetDesktopWindow(), &captureRect);
    }

    int width = captureRect.right - captureRect.left;
    int height = captureRect.bottom - captureRect.top;

    if (width <= 1 || height <= 1) {
        std::cerr << "Invalid region size: width=" << width << ", height=" << height << std::endl;
        ReleaseDC(nullptr, screenDC);
        DeleteDC(memDC);
        return nullptr;
    }

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
    // Validate the IP address using QRegularExpression
    QRegularExpression ipRegex("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b");
    QRegularExpressionMatch match = ipRegex.match(lightIp);
    if (!match.hasMatch()) {
        std::cerr << "Invalid IP Address: " << lightIp.toStdString() << std::endl;
        return; // Exit the function if the IP address is invalid
    }

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
    } else {
        if (dynamicBrightnessCheckbox->isChecked()) {
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
    }

    double adjustedBrightness = (brightness / 100.0) * 255;  // Convert percentage to 0-255 scale

    if (red == 0 && green == 0 && blue == 0) {
        red = green = blue = 1;  // Avoid "off" state being sent as (0,0,0)
        adjustedBrightness = 0;
        lightIsOff = true;
    } else {
        lightIsOff = false;
    }

    // Debugging output
    std::cout << "Using max brightness: " << brightness << std::endl;
    std::cout << "Adjusted brightness sent to WiZ light: " << adjustedBrightness << std::endl;

    QString json = QString("{\"method\":\"setPilot\",\"params\":{\"r\":%1,\"g\":%2,\"b\":%3,\"dimming\":%4}}")
        .arg(red).arg(green).arg(blue).arg(adjustedBrightness);

    udp::endpoint endpoint(boost::asio::ip::make_address(lightIp.toStdString()), 38899);
    udpSocket->send_to(boost::asio::buffer(json.toStdString()), endpoint);
}


// Function to update the color preview box
void UpdateColorDisplay(cv::Scalar color) {
    int red = std::clamp(static_cast<int>(color[2]), 0, 255);   
    int green = std::clamp(static_cast<int>(color[1]), 0, 255);
    int blue = std::clamp(static_cast<int>(color[0]), 0, 255);
    QString style = QString(
        "background-color: rgb(%1, %2, %3);"
        "border-radius: 10px;"
        "border: 1px solid black;").arg(red).arg(green).arg(blue);

    QMetaObject::invokeMethod(colorDisplay, [style]() {
        colorDisplay->setStyleSheet(style);
    }, Qt::QueuedConnection);
}

// Function to update the left color preview box
void UpdateLeftColorDisplay(cv::Scalar color) {
    int red = std::clamp(static_cast<int>(color[2]), 0, 255);   
    int green = std::clamp(static_cast<int>(color[1]), 0, 255);
    int blue = std::clamp(static_cast<int>(color[0]), 0, 255);
    QString style = QString(
        "background-color: rgb(%1, %2, %3);"
        "border-radius: 10px;"
        "border: 1px solid black;").arg(red).arg(green).arg(blue);

    QMetaObject::invokeMethod(leftColorDisplay, [style]() {
        leftColorDisplay->setStyleSheet(style);
    }, Qt::QueuedConnection);
}

// Function to update the right color preview box
void UpdateRightColorDisplay(cv::Scalar color) {
    int red = std::clamp(static_cast<int>(color[2]), 0, 255);   
    int green = std::clamp(static_cast<int>(color[1]), 0, 255);
    int blue = std::clamp(static_cast<int>(color[0]), 0, 255);
    QString style = QString(
        "background-color: rgb(%1, %2, %3);"
        "border-radius: 10px;"
        "border: 1px solid black;").arg(red).arg(green).arg(blue);

    QMetaObject::invokeMethod(rightColorDisplay, [style]() {
        rightColorDisplay->setStyleSheet(style);
    }, Qt::QueuedConnection);
}

void ProcessLeftHalf(const cv::Mat& leftHalfColor, const cv::Mat& leftHalfLBP, double& leftBrightness, cv::Scalar& leftColor) {
    leftColor = getDominantColor(leftHalfColor);
    
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
                    // Wait if paused
                    if (pauseCapture) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }

                    // Capture the screen
                    HBITMAP hBitmap = CaptureScreen();
                    cv::Mat img = HBITMAPToMat(hBitmap);
                    cv::Mat gray;
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                    // Split the image into left and right halves
                    cv::Mat leftHalfColor = img(cv::Rect(0, 0, img.cols / 2, img.rows));
                    cv::Scalar leftColor = getDominantColor(leftHalfColor);
                    double leftAverageBrightness = (cv::mean(leftHalfColor)[0] + cv::mean(leftHalfColor)[1] + cv::mean(leftHalfColor)[2]) / 3.0;

                    // Debugging output
                    std::cout << "Left Color: " << leftColor << std::endl;
                    std::cout << "Left Average Brightness: " << leftAverageBrightness << std::endl;

                    // Update left color display if the left light is defined
                    UpdateLeftColorDisplay(leftColor);

                    // If ambilight mode is enabled, perform additional calculations
                    if (ambilightModeEnabled) {
                        static cv::Mat previousFrame;
                        if (!previousFrame.empty()) {
                            cv::Mat flow = calculateOpticalFlow(previousFrame, gray);
                            cv::resize(gray, gray, cv::Size(), motionImageScale.load(), motionImageScale.load());
                            cv::Mat flowChannels[2];
                            cv::split(flow, flowChannels);
                            cv::Mat magnitude, angle;
                            cv::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);
                            cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
                            magnitude.convertTo(magnitude, CV_8UC1);

                            auto lbpImages = calculateMultiScaleLBP(gray, {1, 2, 3});
                            cv::resize(lbpImages[0], lbpImages[0], cv::Size(), patternImageScale.load(), patternImageScale.load());
                            cv::Mat combinedLBP;
                            cv::addWeighted(lbpImages[0], 0.33, lbpImages[1], 0.33, 0, combinedLBP);
                            cv::addWeighted(combinedLBP, 1.0, lbpImages[2], 0.33, 0, combinedLBP);

                            double motionLeft = cv::mean(magnitude(cv::Rect(0, 0, magnitude.cols / 2, magnitude.rows)))[0];
                            leftAverageBrightness = AdjustBrightnessWithMotion(leftAverageBrightness, motionLeft);
                        }
                        gray.copyTo(previousFrame);
                    }

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
                    // Wait if paused
                    if (pauseCapture) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }

                    // Capture the screen
                    HBITMAP hBitmap = CaptureScreen();
                    cv::Mat img = HBITMAPToMat(hBitmap);
                    cv::Mat gray;
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                    // Split the image into right half
                    cv::Mat rightHalfColor = img(cv::Rect(img.cols / 2, 0, img.cols / 2, img.rows));
                    cv::Scalar rightColor = getDominantColor(rightHalfColor);
                    double rightAverageBrightness = (cv::mean(rightHalfColor)[0] + cv::mean(rightHalfColor)[1] + cv::mean(rightHalfColor)[2]) / 3.0;

                    // Debugging output
                    std::cout << "Right Color: " << rightColor << std::endl;
                    std::cout << "Right Average Brightness: " << rightAverageBrightness << std::endl;

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
                    // Wait if paused
                    if (pauseCapture) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }

                    // Capture the screen
                    HBITMAP hBitmap = CaptureScreen();
                    cv::Mat img = HBITMAPToMat(hBitmap);
                    cv::Scalar dominantColor = getDominantColor(img);
                    double avgBrightness = (cv::mean(img)[0] + cv::mean(img)[1] + cv::mean(img)[2]) / 3.0;

                    // Debugging output
                    std::cout << "Dominant Color: " << dominantColor << std::endl;
                    std::cout << "Average Brightness: " << avgBrightness << std::endl;

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
        // Check if the name exceeds 50 characters
        if (name.length() > 50) {
            QMessageBox::warning(nullptr, "Name Too Long", "The light name cannot exceed 50 characters.");
            return; // Exit the function if the name is too long
        }

        QString ip = QInputDialog::getText(nullptr, "Add Light", "IP Address:", QLineEdit::Normal, "", &ok);
        if (ok && !ip.isEmpty()) {
            // Validate the IP address using QRegularExpression
            QRegularExpression ipRegex("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b");
            QRegularExpressionMatch match = ipRegex.match(ip);
            if (match.hasMatch()) {
                lights.push_back({name, ip});
                lightList->addItem(name);

                // Update combo boxes
                leftLightComboBox->addItem(name);
                rightLightComboBox->addItem(name);
            } else {
                QMessageBox::warning(nullptr, "Invalid IP Address", "The IP address you entered is not valid.");
            }
        }
    }
}


// Function to remove the selected light
void RemoveLight() {
    QListWidgetItem* item = lightList->currentItem();
    if (item) {
        int row = lightList->row(item);  // Get the index of the selected item
        if (row >= 0 && row < lights.size()) {
            // Retrieve the text from the item widget (if using QLabel or other widgets)
            QString lightName;
            QWidget* widget = lightList->itemWidget(item);
            if (widget) {
                QLabel* label = qobject_cast<QLabel*>(widget);  // Cast the widget to QLabel
                if (label) {
                    lightName = label->text();  // Get the text from the QLabel
                }
            } else {
                lightName = item->text();  // If no widget is used, fallback to item text
            }

            // Show confirmation dialog with the light's name
            QMessageBox::StandardButton reply;
            reply = QMessageBox::question(nullptr, "Remove Light?", 
                                          QString("Are you sure you want to remove the light '%1'?").arg(lightName),
                                          QMessageBox::Yes | QMessageBox::No);
            if (reply == QMessageBox::Yes) {
                // Remove the light data from the 'lights' array
                lights.erase(lights.begin() + row);
                
                // Remove the item from the light list UI
                delete lightList->takeItem(row);

                // Update combo boxes by removing the corresponding item
                leftLightComboBox->removeItem(row + 1);  // +1 to account for "None" option
                rightLightComboBox->removeItem(row + 1);
            }
        }
    }
}

void DiscoverLights() {
    std::thread([]() {
        try {
            boost::asio::io_context context;
            udp::socket socket(context, udp::endpoint(udp::v4(), 0));
            socket.set_option(boost::asio::socket_base::broadcast(true));

            const std::string discoveryMessage = R"({"method":"getSystemConfig","params":{}})";
            udp::endpoint broadcastEndpoint(boost::asio::ip::address_v4::broadcast(), 38899);

            socket.send_to(boost::asio::buffer(discoveryMessage), broadcastEndpoint);

            char buffer[1024];
            udp::endpoint senderEndpoint;

            std::unordered_map<std::string, std::string> systemConfigs;

            while (true) {
                std::memset(buffer, 0, sizeof(buffer));
                size_t length = socket.receive_from(boost::asio::buffer(buffer), senderEndpoint);

                std::string response(buffer, length);
                nlohmann::json jsonResponse = nlohmann::json::parse(response);

                if (jsonResponse.contains("result") && jsonResponse["result"].contains("moduleName")) {
                    std::string ipAddress = senderEndpoint.address().to_string();
                    std::string lightName = jsonResponse["result"]["moduleName"];

                    systemConfigs[ipAddress] = lightName;

                    std::string stateMessage = R"({"method":"getPilot","params":{}})";
                    udp::endpoint lightEndpoint(boost::asio::ip::make_address(ipAddress), 38899);
                    socket.send_to(boost::asio::buffer(stateMessage), lightEndpoint);
                } else if (jsonResponse.contains("method") && jsonResponse["method"] == "getPilot") {
                    std::string ipAddress = senderEndpoint.address().to_string();
                    QString lightName = QString::fromStdString(systemConfigs[ipAddress]);

                    QString powerState = "UNKNOWN";
                    if (jsonResponse.contains("result")) {
                        if (jsonResponse["result"].contains("state") && jsonResponse["result"]["state"].is_boolean()) {
                            powerState = jsonResponse["result"]["state"].get<bool>() ? "ON" : "OFF";
                        } else if (jsonResponse["result"].contains("state") && jsonResponse["result"]["state"].is_number_integer()) {
                            powerState = jsonResponse["result"]["state"].get<int>() == 1 ? "ON" : "OFF";
                        }
                    }

                    QString mode = "Unknown";
                    QString displayString;
                    if (jsonResponse["result"].contains("sceneId") && jsonResponse["result"]["sceneId"].is_number_integer()) {
                        int sceneId = jsonResponse["result"]["sceneId"].get<int>();
                        if (sceneId == 0) {
                            if (jsonResponse["result"].contains("r") && jsonResponse["result"].contains("g") && jsonResponse["result"].contains("b")) {
                                int r = jsonResponse["result"]["r"].get<int>();
                                int g = jsonResponse["result"]["g"].get<int>();
                                int b = jsonResponse["result"]["b"].get<int>();
                                QString rgbText = QString("<span style='color: rgb(%1, %2, %3);'>RGB (%1, %2, %3)</span>").arg(r).arg(g).arg(b);
                                displayString = QString("%1 - %2 - %3 - %4").arg(powerState).arg(lightName).arg(QString::fromStdString(ipAddress)).arg(rgbText);
                            }
                        } else {
                            mode = QString::fromStdString(GetSceneName(sceneId));
                            displayString = QString("%1 - %2 - %3 - %4").arg(powerState).arg(lightName).arg(QString::fromStdString(ipAddress)).arg(mode);
                        }
                    } else if (jsonResponse["result"].contains("r")) {
                        mode = "Color";
                        displayString = QString("%1 - %2 - %3 - %4").arg(powerState).arg(lightName).arg(QString::fromStdString(ipAddress)).arg(mode);
                    } else {
                        displayString = QString("%1 - %2 - %3 - %4").arg(powerState).arg(lightName).arg(QString::fromStdString(ipAddress)).arg(mode);
                    }

                    bool exists = false;
                    for (const auto& light : lights) {
                        if (light.ipAddress == QString::fromStdString(ipAddress)) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        Light newLight{displayString, QString::fromStdString(ipAddress)};
                        lights.push_back(newLight);

                        QMetaObject::invokeMethod(QApplication::instance(), [=]() {
                            QListWidgetItem* item = new QListWidgetItem();
                            QLabel* label = new QLabel(displayString);
                            label->setTextFormat(Qt::RichText);
                            lightList->addItem(item);
                            lightList->setItemWidget(item, label);

                            // Remove rich text formatting for combo boxes
                            QString plainText = removeHtmlTags(displayString);

                            leftLightComboBox->addItem(plainText);
                            rightLightComboBox->addItem(plainText);
                        }, Qt::QueuedConnection);
                    }
                } else {
                    std::cerr << "Unrecognized JSON response: " << response << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during discovery: " << e.what() << std::endl;
        }
    }).detach();
}

void RenameLight() {
    QListWidgetItem* selectedItem = lightList->currentItem();
    if (!selectedItem) return;

    bool ok;
    QString currentText = selectedItem->text();
    QString newName = QInputDialog::getText(nullptr, "Rename Light", "Enter new name:", QLineEdit::Normal, currentText, &ok);

    if (ok && !newName.isEmpty()) {
        if (newName.length() > 50) {
            QMessageBox::warning(nullptr, "Name Too Long", "The light name cannot exceed 50 characters.");
            return;
        }

        int index = lightList->row(selectedItem);

        lights[index].name = newName;

        lightList->removeItemWidget(selectedItem);
        selectedItem->setText(newName);

        leftLightComboBox->setItemText(index + 1, newName); // Update combo box display
        rightLightComboBox->setItemText(index + 1, newName);
    }
}


void SelectScreenRegion() {
    if (capturing) {
        pauseCapture = true; // Pause capturing
    }

    HINSTANCE hInstance = GetModuleHandle(nullptr);

    WNDCLASS wndClass = {};
    wndClass.lpfnWndProc = [](HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) -> LRESULT {
        static POINT startPoint, endPoint;
        static HBRUSH hBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
        static HDC hdcMem = nullptr;
        static HBITMAP hbmMem = nullptr;
        static HGDIOBJ hOld = nullptr;

        switch (msg) {
        case WM_CREATE: {
            RECT rect;
            GetClientRect(hwnd, &rect);
            HDC hdc = GetDC(hwnd);
            hdcMem = CreateCompatibleDC(hdc);
            hbmMem = CreateCompatibleBitmap(hdc, rect.right, rect.bottom);
            hOld = SelectObject(hdcMem, hbmMem);
            ReleaseDC(hwnd, hdc);
            return 0;
        }
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

                // Adjust coordinates for the selected monitor
                int selectedMonitorIndex = monitorComboBox->currentIndex();
                if (selectedMonitorIndex >= 0 && selectedMonitorIndex < monitors.size()) {
                    RECT monitorBounds = monitors[selectedMonitorIndex].bounds;
                    selectedRegion.left += monitorBounds.left;
                    selectedRegion.top += monitorBounds.top;
                    selectedRegion.right += monitorBounds.left;
                    selectedRegion.bottom += monitorBounds.top;
                }

                int width = selectedRegion.right - selectedRegion.left;
                int height = selectedRegion.bottom - selectedRegion.top;
                if (width <= 1 || height <= 1) {
                    MessageBox(hwnd, "Selected region is inavlid or too small. Please select a valid region or larger region.", "Error: Invalid Selection", MB_OK | MB_ICONERROR);
                    regionSelected = false;
                } else {
                    // Resume capturing if the region is valid
                    if (capturing) {
                        pauseCapture = false;
                    }
                }

                DestroyWindow(hwnd);
            }
            return 0;
        case WM_PAINT: {
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
            return 0;
        }
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
    wndClass.style = CS_HREDRAW | CS_VREDRAW;
    RegisterClass(&wndClass);

    // Set the window position and size based on the selected monitor
    int selectedMonitorIndex = monitorComboBox->currentIndex();
    if (selectedMonitorIndex >= 0 && selectedMonitorIndex < monitors.size()) {
        RECT monitorBounds = monitors[selectedMonitorIndex].bounds;
        HWND hwnd = CreateWindowEx(
            WS_EX_TOPMOST | WS_EX_LAYERED,
            "RegionSelector",
            "Select Screen Region",
            WS_POPUP | WS_VISIBLE,
            monitorBounds.left, monitorBounds.top,
            monitorBounds.right - monitorBounds.left, 
            monitorBounds.bottom - monitorBounds.top,
            nullptr, nullptr, hInstance, nullptr
        );

        SetLayeredWindowAttributes(hwnd, 0, 128, LWA_ALPHA); // Set alpha to 128 for partial transparency

        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    } else {
        MessageBox(nullptr, "No monitor selected or invalid monitor index.", "Error", MB_OK | MB_ICONERROR);
    }
}



const int DEFAULT_DARKNESS_THRESHOLD = 30;
const int DEFAULT_MAX_BRIGHTNESS = 100;
const int DEFAULT_COLOR_BOOST_INTENSITY = 20;
const double DEFAULT_MAIN_IMAGE_SCALE = 0.50;
const double DEFAULT_MOTION_IMAGE_SCALE = 0.50;
const double DEFAULT_PATTERN_IMAGE_SCALE = 0.50;
const bool DEFAULT_ADJUST_BRIGHTNESS = false;
const bool DEFAULT_COLOR_BOOST_ENABLED = false;
const bool DEFAULT_AMBILIGHT_MODE_ENABLED = false;

void ResetToDefaults() {
    // Reset all settings to their default values
    darknessThreshold = DEFAULT_DARKNESS_THRESHOLD;
    maxBrightness = DEFAULT_MAX_BRIGHTNESS;
    colorBoostIntensity = DEFAULT_COLOR_BOOST_INTENSITY;
    adjustBrightness = DEFAULT_ADJUST_BRIGHTNESS;
    colorBoostEnabled = DEFAULT_COLOR_BOOST_ENABLED;
    ambilightModeEnabled = DEFAULT_AMBILIGHT_MODE_ENABLED;
    mainImageScale = DEFAULT_MAIN_IMAGE_SCALE;
    motionImageScale = DEFAULT_MOTION_IMAGE_SCALE;
    patternImageScale = DEFAULT_PATTERN_IMAGE_SCALE;
    lights.clear();

    // Update the UI elements to reflect the default settings
    brightnessCheckbox->setChecked(adjustBrightness);
    darknessSlider->setValue(darknessThreshold);
    maxBrightnessSlider->setValue(maxBrightness);
    colorBoostCheckbox->setChecked(colorBoostEnabled);
    colorBoostSlider->setValue(colorBoostIntensity);
    darknessThresholdSlider->setValue(darknessThreshold);
    dynamicBrightnessCheckbox->setChecked(adjustBrightness);
    
    // Ensure the sliders' values reflect percentages
    mainImageScaleSlider->setValue(static_cast<int>(mainImageScale * 100));
    motionImageScaleSlider->setValue(static_cast<int>(motionImageScale * 100));
    patternImageScaleSlider->setValue(static_cast<int>(patternImageScale * 100));

    // Clear the light list
    lightList->clear();
    leftLightComboBox->clear();
    rightLightComboBox->clear();
    leftLightComboBox->addItem("None");
    rightLightComboBox->addItem("None");

    // Update current preset label
    if (currentPresetLabel) {
        currentPresetLabel->setText("No preset loaded");
    }
}

// Function to save settings to a JSON file
void SaveSettings() {
    nlohmann::json settings;

    // Save darkness threshold, color boost intensity, and max brightness
    settings["darknessThreshold"] = darknessThreshold.load();
    settings["colorBoostIntensity"] = colorBoostIntensity.load();
    settings["darknessThresholdValue"] = darknessThresholdSlider->value(); // Save the darkness threshold slider value
    settings["dynamicBrightnessEnabled"] = dynamicBrightnessCheckbox->isChecked(); // Save dynamic brightness state
    settings["mainImageScale"] = mainImageScale.load();
    settings["motionImageScale"] = motionImageScale.load();
    settings["patternImageScale"] = patternImageScale.load();
    settings["maxBrightness"] = maxBrightness.load(); // Save max brightness

    // Save whether darkness adjustment and color boost are enabled
    settings["adjustBrightness"] = adjustBrightness.load();
    settings["colorBoostEnabled"] = colorBoostEnabled.load();
    // Save if ambilight mode is enabled
    settings["ambilightModeEnabled"] = ambilightModeEnabled.load();

    // Save lights with RGB values
    settings["lights"] = nlohmann::json::array();
    for (const Light& light : lights) {
        nlohmann::json lightJson = {
            {"name", light.name.toStdString()},
            {"ipAddress", light.ipAddress.toStdString()}
        };
        if (light.name.contains("RGB (")) {
            // Extract and save RGB values without altering the original name
            QStringList parts = light.name.split(" - ");
            QString baseName = parts.mid(0, parts.size() - 1).join(" - ");
            QString rgbPart = parts.last().split("RGB (").last().split(")").first();
            QStringList rgbValues = rgbPart.split(", ");
            if (rgbValues.size() == 3) {
                lightJson["r"] = rgbValues[0].toInt();
                lightJson["g"] = rgbValues[1].toInt();
                lightJson["b"] = rgbValues[2].toInt();
                lightJson["name"] = baseName.toStdString();
            }
        }
        settings["lights"].push_back(lightJson);
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

        // Load darkness threshold, color boost intensity, and max brightness
        darknessThreshold = settings.value("darknessThreshold", DEFAULT_DARKNESS_THRESHOLD);
        colorBoostIntensity = settings.value("colorBoostIntensity", DEFAULT_COLOR_BOOST_INTENSITY);
        int darknessThresholdValue = settings.value("darknessThresholdValue", 20); // Load the darkness threshold slider value
        bool dynamicBrightnessEnabled = settings.value("dynamicBrightnessEnabled", false); // Load dynamic brightness state
        mainImageScale = settings.value("mainImageScale", 0.5);  // Default to 0.5
        motionImageScale = settings.value("motionImageScale", 0.5);
        patternImageScale = settings.value("patternImageScale", 0.5);
        maxBrightness = settings.value("maxBrightness", 100); // Load max brightness

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
            QString lightName = QString::fromStdString(light["name"]);
            QString displayName = lightName; // Keep original light name for display purposes
            
            if (light.contains("r") && light.contains("g") && light.contains("b")) {
                // Format RGB text
                QString rgbText = QString("<span style='color: rgb(%1, %2, %3);'>RGB (%1, %2, %3)</span>")
                                  .arg(light["r"].get<int>())
                                  .arg(light["g"].get<int>())
                                  .arg(light["b"].get<int>());
                displayName += " - " + rgbText; // Use displayName with HTML tags for light list
            }

            // Add cleaned light name (without HTML) to combo boxes
            leftLightComboBox->addItem(removeHtmlTags(displayName));
            rightLightComboBox->addItem(removeHtmlTags(displayName));

            // Store light info with HTML in lights list
            lights.push_back({displayName, QString::fromStdString(light["ipAddress"])}); // Store full name with HTML
            
            // Add to the list widget with rich text
            QListWidgetItem* item = new QListWidgetItem();
            QLabel* label = new QLabel(displayName);
            label->setTextFormat(Qt::RichText); // Set text format to Rich Text
            lightList->addItem(item);
            lightList->setItemWidget(item, label);
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
        if (maxBrightnessSlider) maxBrightnessSlider->setValue(maxBrightness); // Update max brightness slider
        
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



// Ensure the presets directory exists
void EnsurePresetsDirectoryExists() {
    QDir dir(presetsDirectory);
    if (!dir.exists()) {
        dir.mkpath(".");
    }
}

void SavePresetToFile(const Preset& preset) {
    EnsurePresetsDirectoryExists();
    nlohmann::json presetJson = {
        {"name", preset.name.toStdString()},
        {"darknessThreshold", preset.darknessThreshold},
        {"colorBoostIntensity", preset.colorBoostIntensity},
        {"adjustBrightness", preset.adjustBrightness},
        {"colorBoostEnabled", preset.colorBoostEnabled},
        {"ambilightModeEnabled", preset.ambilightModeEnabled},
        {"mainImageScale", preset.mainImageScale},
        {"motionImageScale", preset.motionImageScale},
        {"patternImageScale", preset.patternImageScale},
        {"maxBrightness", maxBrightness.load()},
        {"dynamicBrightnessEnabled", dynamicBrightnessCheckbox->isChecked()} // Save dynamic brightness state
    };

    for (const auto& light : preset.lights) {
        nlohmann::json lightJson = {
            {"name", light.name.toStdString()},
            {"ipAddress", light.ipAddress.toStdString()}
        };
        if (light.name.contains("RGB (")) {
            // Extract and save RGB values
            QString rgbPart = light.name.split("RGB (").last().split(")").first();
            QStringList rgbValues = rgbPart.split(", ");
            if (rgbValues.size() == 3) {
                lightJson["r"] = rgbValues[0].toInt();
                lightJson["g"] = rgbValues[1].toInt();
                lightJson["b"] = rgbValues[2].toInt();
                // Remove the RGB part from the name to avoid duplication
                lightJson["name"] = light.name.split("RGB (").first().trimmed().toStdString();
            }
        }
        presetJson["lights"].push_back(lightJson);
    }

    std::ofstream file(presetsDirectory.toStdString() + preset.name.toStdString() + ".json");
    file << presetJson.dump(4);
}

Preset LoadPresetFromFile(const QString& presetName) {
    std::ifstream file(presetsDirectory.toStdString() + presetName.toStdString() + ".json");
    nlohmann::json presetJson;
    file >> presetJson;

    Preset preset;
    preset.name = QString::fromStdString(presetJson["name"]);
    preset.darknessThreshold = presetJson["darknessThreshold"];
    preset.colorBoostIntensity = presetJson["colorBoostIntensity"];
    preset.adjustBrightness = presetJson["adjustBrightness"];
    preset.colorBoostEnabled = presetJson["colorBoostEnabled"];
    preset.ambilightModeEnabled = presetJson["ambilightModeEnabled"];
    preset.mainImageScale = presetJson["mainImageScale"];
    preset.motionImageScale = presetJson["motionImageScale"];
    preset.patternImageScale = presetJson["patternImageScale"];
    preset.maxBrightness = presetJson.value("maxBrightness", 100);
    bool dynamicBrightnessEnabled = presetJson.value("dynamicBrightnessEnabled", false); // Load dynamic brightness state

    std::cout << "Loaded dynamicBrightnessEnabled: " << dynamicBrightnessEnabled << std::endl;

    for (const auto& lightJson : presetJson["lights"]) {
        QString lightName = QString::fromStdString(lightJson["name"]);
        if (lightJson.contains("r") && lightJson.contains("g") && lightJson.contains("b")) {
            // Format RGB text
            QString rgbText = QString("<span style='color: rgb(%1, %2, %3);'>RGB (%1, %2, %3)</span>")
                              .arg(lightJson["r"].get<int>())
                              .arg(lightJson["g"].get<int>())
                              .arg(lightJson["b"].get<int>());
            lightName += " - " + rgbText;
        }
        Light light;
        light.name = lightName;
        light.ipAddress = QString::fromStdString(lightJson["ipAddress"]);
        preset.lights.push_back(light);
    }

    // Update the dynamicBrightnessCheckbox state
    QMetaObject::invokeMethod(dynamicBrightnessCheckbox, [dynamicBrightnessEnabled]() {
        dynamicBrightnessCheckbox->setChecked(dynamicBrightnessEnabled);
        std::cout << "Set dynamicBrightnessCheckbox to: " << dynamicBrightnessEnabled << std::endl;
    }, Qt::QueuedConnection);

    return preset;
}


void DeletePresetFile(const QString& presetName) {
    QFile::remove(presetsDirectory + presetName + ".json");
}

void SavePreset(const QString& presetName) {
    Preset preset;
    preset.name = presetName;
    preset.darknessThreshold = darknessThreshold.load();
    preset.colorBoostIntensity = colorBoostIntensity.load();
    preset.adjustBrightness = adjustBrightness.load();
    preset.colorBoostEnabled = colorBoostEnabled.load();
    preset.ambilightModeEnabled = ambilightModeEnabled.load();
    preset.mainImageScale = mainImageScale.load();
    preset.motionImageScale = motionImageScale.load();
    preset.patternImageScale = patternImageScale.load();
    preset.lights = lights;

    SavePresetToFile(preset);
}

void LoadPreset(const QString& presetName) {
    std::cout << "Loading preset: " << presetName.toStdString() << std::endl;

    Preset preset = LoadPresetFromFile(presetName);
    std::cout << "Preset loaded from file: " << presetName.toStdString() << std::endl;

    darknessThreshold = preset.darknessThreshold;
    colorBoostIntensity = preset.colorBoostIntensity;
    adjustBrightness = preset.adjustBrightness;
    colorBoostEnabled = preset.colorBoostEnabled;
    ambilightModeEnabled = preset.ambilightModeEnabled;
    mainImageScale = preset.mainImageScale;
    motionImageScale = preset.motionImageScale;
    patternImageScale = preset.patternImageScale;
    maxBrightness = preset.maxBrightness;
    lights = preset.lights;

    std::cout << "Updating UI elements" << std::endl;

    auto updateUiElements = [preset]() {
        brightnessCheckbox->setChecked(preset.adjustBrightness);
        std::cout << "Updated brightnessCheckbox" << std::endl;

        darknessSlider->setValue(preset.darknessThreshold);
        std::cout << "Updated darknessSlider" << std::endl;

        colorBoostCheckbox->setChecked(preset.colorBoostEnabled);
        std::cout << "Updated colorBoostCheckbox" << std::endl;

        colorBoostSlider->setValue(preset.colorBoostIntensity);
        std::cout << "Updated colorBoostSlider" << std::endl;

        darknessThresholdSlider->setValue(preset.darknessThreshold);
        std::cout << "Updated darknessThresholdSlider" << std::endl;

        dynamicBrightnessCheckbox->setChecked(preset.dynamicBrightnessEnabled); // Ensure correct state
        std::cout << "Updated dynamicBrightnessCheckbox" << std::endl;

        mainImageScaleSlider->setValue(static_cast<int>(preset.mainImageScale * 100));
        std::cout << "Updated mainImageScaleSlider" << std::endl;

        motionImageScaleSlider->setValue(static_cast<int>(preset.motionImageScale * 100));
        std::cout << "Updated motionImageScaleSlider" << std::endl;

        patternImageScaleSlider->setValue(static_cast<int>(preset.patternImageScale * 100));
        std::cout << "Updated patternImageScaleSlider" << std::endl;

        maxBrightnessSlider->setValue(preset.maxBrightness); // Update max brightness slider
        std::cout << "Updated maxBrightnessSlider to: " << preset.maxBrightness << std::endl;
    };
    QMetaObject::invokeMethod(QApplication::instance(), updateUiElements, Qt::QueuedConnection);

    // Update light list
    lightList->clear();
    leftLightComboBox->clear();
    rightLightComboBox->clear();
    leftLightComboBox->addItem("None");
    rightLightComboBox->addItem("None");

    for (const auto& light : lights) {
        QListWidgetItem* item = new QListWidgetItem();
        QLabel* label = new QLabel(light.name);
        label->setTextFormat(Qt::RichText); // Set text format to Rich Text
        lightList->addItem(item);
        lightList->setItemWidget(item, label);
        leftLightComboBox->addItem(light.name);
        rightLightComboBox->addItem(light.name);
    }

    std::cout << "Light list updated" << std::endl;

    // Ensure currentPresetLabel is not null before updating
    if (currentPresetLabel) {
        QMetaObject::invokeMethod(currentPresetLabel, [presetName]() {
            currentPresetLabel->setText("Current Preset: " + presetName);
            std::cout << "Updated currentPresetLabel" << std::endl;
        }, Qt::QueuedConnection);
    }

    std::cout << "Preset loaded successfully" << std::endl;
}


void DeletePreset(const QString& presetName) {
    DeletePresetFile(presetName);
}

void RenamePreset(const QString& oldName, const QString& newName) {
    std::string oldFilename = presetsDirectory.toStdString() + oldName.toStdString() + ".json";
    std::string newFilename = presetsDirectory.toStdString() + newName.toStdString() + ".json";

    // Rename the file
    std::rename(oldFilename.c_str(), newFilename.c_str());

    // Also update the name in the presets list
    for (auto& preset : presets) {
        if (preset.name == oldName) {
            preset.name = newName;
            break;
        }
    }
}


void SetupUI(QVBoxLayout* layout) {
    // Add start and stop buttons
    QPushButton* savePresetButton = new QPushButton("Save Preset");
    QPushButton* loadPresetButton = new QPushButton("Load Preset");

    QPushButton* renamePresetButton = new QPushButton("Rename Preset");
    QPushButton* deletePresetButton = new QPushButton("Delete Preset");

    QPushButton* startButton = new QPushButton("Start");
    QPushButton* stopButton = new QPushButton("Stop");
    layout->addWidget(startButton);
    layout->addWidget(stopButton);

    // Create a horizontal layout for the region selection and preview buttons
    QHBoxLayout* regionPreviewLayout = new QHBoxLayout;

    // Add region selection button
    QPushButton* selectRegionButton = new QPushButton("Select Region");
    regionPreviewLayout->addWidget(selectRegionButton);

    // Add preview button
    QPushButton* previewButton = new QPushButton("Preview");
    regionPreviewLayout->addWidget(previewButton);

    // Add the horizontal layout to the main layout
    layout->addLayout(regionPreviewLayout);

    // Create a scroll area and a widget for the scroll area
    QScrollArea* scrollArea = new QScrollArea;
    QWidget* scrollWidget = new QWidget;
    QVBoxLayout* scrollLayout = new QVBoxLayout(scrollWidget);

    // Add the monitor selection above the color display layout
    QLabel* monitorLabel = new QLabel("Select Monitor:");
    monitorComboBox = new QComboBox();
    scrollLayout->addWidget(monitorLabel);
    scrollLayout->addWidget(monitorComboBox);

    // Create a vertical layout for each color display box
    QVBoxLayout* leftColorLayout = new QVBoxLayout;
    QVBoxLayout* dominantColorLayout = new QVBoxLayout;
    QVBoxLayout* rightColorLayout = new QVBoxLayout;

    // Initialize and add color display for left color
    leftColorLabel = new QLabel("Left Color:");
    leftColorDisplay = new QLabel;
    leftColorDisplay->setFixedSize(100, 50);
    leftColorDisplay->setStyleSheet("background-color: rgb(0, 0, 0); border-radius: 10px; border: 1px solid black;");
    leftColorLayout->addWidget(leftColorLabel);
    leftColorLayout->addWidget(leftColorDisplay);

    // Initialize and add color display for dominant color
    colorLabel = new QLabel("Dominant Color:");
    colorDisplay = new QLabel;
    colorDisplay->setFixedSize(100, 50);
    colorDisplay->setStyleSheet("background-color: rgb(0, 0, 0); border-radius: 10px; border: 1px solid black;");
    dominantColorLayout->addWidget(colorLabel);
    dominantColorLayout->addWidget(colorDisplay);

    // Initialize and add color display for right color
    rightColorLabel = new QLabel("Right Color:");
    rightColorDisplay = new QLabel;
    rightColorDisplay->setFixedSize(100, 50);
    rightColorDisplay->setStyleSheet("background-color: rgb(0, 0, 0); border-radius: 10px; border: 1px solid black;");
    rightColorLayout->addWidget(rightColorLabel);
    rightColorLayout->addWidget(rightColorDisplay);

    // Create a horizontal layout to hold the vertical layouts
    QHBoxLayout* colorDisplayLayout = new QHBoxLayout;
    colorDisplayLayout->addLayout(leftColorLayout);
    colorDisplayLayout->addLayout(dominantColorLayout);
    colorDisplayLayout->addLayout(rightColorLayout);

    // Add the horizontal layout to the scroll layout
    scrollLayout->addLayout(colorDisplayLayout);

    // Add list widget for lights
    lightsLabel = new QLabel("WiZ Light List:");
    lightList = new QListWidget;
    lightList->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    scrollLayout->addWidget(lightsLabel);
    scrollLayout->addWidget(lightList);
    const int baseMinimumWidth = 310; // Adjust the minimum width of the light list

    // Add add/remove buttons for lights in a horizontal layout
    QPushButton* addButton = new QPushButton("Add Light");
    QPushButton* removeButton = new QPushButton("Remove Light");
    QHBoxLayout* addRemoveLayout = new QHBoxLayout;
    addRemoveLayout->addWidget(addButton);
    addRemoveLayout->addWidget(removeButton);
    scrollLayout->addLayout(addRemoveLayout);

    // Add discover/rename buttons for lights in a horizontal layout
    QPushButton* discoverButton = new QPushButton("Discover Lights");
    QPushButton* renameButton = new QPushButton("Rename Light");
    QHBoxLayout* discoverRenameLayout = new QHBoxLayout;
    discoverRenameLayout->addWidget(discoverButton);
    discoverRenameLayout->addWidget(renameButton);
    scrollLayout->addLayout(discoverRenameLayout);


    scrollLayout->addSpacing(10); // spacing between gui elements

    // Create vertical layouts for left and right light selection
    QVBoxLayout* leftLightLayout = new QVBoxLayout;
    leftLightLayout->addWidget(new QLabel("Select Light for Left Half:"));
    leftLightComboBox = new QComboBox();
    leftLightComboBox->addItem("None");
    leftLightLayout->addWidget(leftLightComboBox);

    QVBoxLayout* rightLightLayout = new QVBoxLayout;
    rightLightLayout->addWidget(new QLabel("Select Light for Right Half:"));
    rightLightComboBox = new QComboBox();
    rightLightComboBox->addItem("None");
    rightLightLayout->addWidget(rightLightComboBox);

    // Create a horizontal layout to hold the vertical layouts
    QHBoxLayout* lightSelectionLayout = new QHBoxLayout;
    lightSelectionLayout->addLayout(leftLightLayout);
    lightSelectionLayout->addLayout(rightLightLayout);

    // Add the horizontal layout to the scroll layout
    scrollLayout->addLayout(lightSelectionLayout);

    scrollLayout->addSpacing(20); // spacing between gui elements

    // Initialize the global currentPresetLabel
    currentPresetLabel = new QLabel("No preset loaded");
    layout->addWidget(currentPresetLabel);

    // Add preset buttons in a horizontal layout
    QHBoxLayout* presetButtonsLayout = new QHBoxLayout;
    presetButtonsLayout->addWidget(savePresetButton);
    presetButtonsLayout->addWidget(loadPresetButton);
    presetButtonsLayout->addWidget(renamePresetButton);
    presetButtonsLayout->addWidget(deletePresetButton);
    layout->addLayout(presetButtonsLayout);

    // Add brightness adjustment controls
    brightnessCheckbox = new QCheckBox("Adjust Brightness Based on Darkness");
    scrollLayout->addWidget(brightnessCheckbox);

    // Add the label for darkness threshold
    QLabel* darknessSliderLabel = new QLabel("Darkness Threshold Intensity:");
    scrollLayout->addWidget(darknessSliderLabel);

    darknessSlider = new QSlider(Qt::Horizontal);
    darknessSlider->setRange(0, 100);
    scrollLayout->addWidget(darknessSlider);

    // Create the value label and position it above the slider
    QLabel* darknessSliderValueLabel = new QLabel("20", scrollWidget);
    darknessSliderValueLabel->hide(); // Ensure it's hidden initially
    darknessSliderValueLabel->setStyleSheet("background-color: white; border: 1px solid black; border-radius: 4px;");
    darknessSliderValueLabel->setAlignment(Qt::AlignCenter);
    darknessSliderValueLabel->setFixedSize(30, 20);

    scrollLayout->addSpacing(15); // spacing between gui elements

    // Add the label for darkness threshold
    QLabel* darknessThresholdLabel = new QLabel("Darkness Threshold:");
    scrollLayout->addWidget(darknessThresholdLabel);

    // Create and set up the slider
    darknessThresholdSlider = new QSlider(Qt::Horizontal);
    darknessThresholdSlider->setRange(0, 255);
    darknessThresholdSlider->setValue(20); // Default value
    scrollLayout->addWidget(darknessThresholdSlider);

    // Create the value label and position it above the slider
    QLabel* darknessThresholdValueLabel = new QLabel("20", scrollWidget);
    darknessThresholdValueLabel->hide(); // Ensure it's hidden initially
    darknessThresholdValueLabel->setStyleSheet("background-color: white; border: 1px solid black; border-radius: 4px;");
    darknessThresholdValueLabel->setAlignment(Qt::AlignCenter);
    darknessThresholdValueLabel->setFixedSize(30, 20);

    scrollLayout->addSpacing(15); // spacing between gui elements

    // Add maximum brightness slider
    QLabel* maxBrightnessLabel = new QLabel("Maximum Brightness:");
    scrollLayout->addWidget(maxBrightnessLabel);
    maxBrightnessSlider = new QSlider(Qt::Horizontal); // Initialize the global variable
    maxBrightnessSlider->setRange(0, 255); // Set the range to 0-100
    maxBrightnessSlider->setValue(100); // Default value is 100%
    scrollLayout->addWidget(maxBrightnessSlider);

    // Add for maxBrightnessSlider value label
    QLabel* maxBrightnessValueLabel = new QLabel("100", scrollWidget);
    maxBrightnessValueLabel->hide();
    maxBrightnessValueLabel->setStyleSheet("background-color: white; border: 1px solid black; border-radius: 4px;");
    maxBrightnessValueLabel->setAlignment(Qt::AlignCenter);
    maxBrightnessValueLabel->setFixedSize(30, 20);

    scrollLayout->addSpacing(20); // spacing between gui elements

    // Add dynamic brightness checkbox
    dynamicBrightnessCheckbox = new QCheckBox("Enable Dynamic Brightness");
    scrollLayout->addWidget(dynamicBrightnessCheckbox);

    // Add color boost controls
    colorBoostSlider = new QSlider(Qt::Horizontal);
    colorBoostSlider->setRange(0, 100);
    scrollLayout->addWidget(new QLabel("Color Boost Intensity"));
    scrollLayout->addWidget(colorBoostSlider);

    // Add for colorBoostSlider
    QLabel* colorBoostValueLabel = new QLabel("50", scrollWidget);
    colorBoostValueLabel->hide();
    colorBoostValueLabel->setStyleSheet("background-color: white; border: 1px solid black; border-radius: 4px;");
    colorBoostValueLabel->setAlignment(Qt::AlignCenter);
    colorBoostValueLabel->setFixedSize(30, 20);

    scrollLayout->addSpacing(20); // spacing between gui elements

    colorBoostCheckbox = new QCheckBox("Enable Color Boost");
    scrollLayout->addWidget(colorBoostCheckbox);

    // Add scaling sliders
    QLabel* mainImageScaleLabel = new QLabel("Main Image Scaling:");
    mainImageScaleSlider = new QSlider(Qt::Horizontal); // Assign to the global pointer
    mainImageScaleSlider->setRange(10, 100);  // Scale from 0.1 to 1.0 (multiplied by 100 for UI)
    mainImageScaleSlider->setValue(static_cast<int>(mainImageScale.load() * 100));
    scrollLayout->addWidget(mainImageScaleLabel);
    scrollLayout->addWidget(mainImageScaleSlider);

    // Add for mainImageScaleSlider
    QLabel* mainImageScaleValueLabel = new QLabel("50", scrollWidget);
    mainImageScaleValueLabel->hide();
    mainImageScaleValueLabel->setStyleSheet("background-color: white; border: 1px solid black; border-radius: 4px;");
    mainImageScaleValueLabel->setAlignment(Qt::AlignCenter);
    mainImageScaleValueLabel->setFixedSize(30, 20);

    scrollLayout->addSpacing(15); // spacing between gui elements

    QLabel* motionImageScaleLabel = new QLabel("Motion Image Scaling:");
    motionImageScaleSlider = new QSlider(Qt::Horizontal); // Assign to the global pointer
    motionImageScaleSlider->setRange(10, 100);
    motionImageScaleSlider->setValue(static_cast<int>(motionImageScale.load() * 100));
    scrollLayout->addWidget(motionImageScaleLabel);
    scrollLayout->addWidget(motionImageScaleSlider);

    // Add for motionImageScaleSlider
    QLabel* motionImageScaleValueLabel = new QLabel("50", scrollWidget);
    motionImageScaleValueLabel->hide();
    motionImageScaleValueLabel->setStyleSheet("background-color: white; border: 1px solid black; border-radius: 4px;");
    motionImageScaleValueLabel->setAlignment(Qt::AlignCenter);
    motionImageScaleValueLabel->setFixedSize(30, 20);

    scrollLayout->addSpacing(15); // spacing between gui elements

    QLabel* patternImageScaleLabel = new QLabel("Pattern Image Scaling:");
    patternImageScaleSlider = new QSlider(Qt::Horizontal); // Assign to the global pointer
    patternImageScaleSlider->setRange(10, 100);
    patternImageScaleSlider->setValue(static_cast<int>(patternImageScale.load() * 100));
    scrollLayout->addWidget(patternImageScaleLabel);
    scrollLayout->addWidget(patternImageScaleSlider);

    // Add for patternImageScaleSlider
    QLabel* patternImageScaleValueLabel = new QLabel("50", scrollWidget);
    patternImageScaleValueLabel->hide();
    patternImageScaleValueLabel->setStyleSheet("background-color: white; border: 1px solid black; border-radius: 4px;");
    patternImageScaleValueLabel->setAlignment(Qt::AlignCenter);
    patternImageScaleValueLabel->setFixedSize(30, 20);


    scrollLayout->addSpacing(20); // spacing between gui elements

    ambilightModeCheckbox = new QCheckBox("Enable Ambilight Mode");
    scrollLayout->addWidget(ambilightModeCheckbox);

    scrollLayout->addSpacing(15); // spacing between gui elements

    // Add reset to defaults button
    QPushButton* resetButton = new QPushButton("Reset to Defaults");
    scrollLayout->addWidget(resetButton);

    // Set the layout for the scroll widget and add it to the scroll area
    scrollWidget->setLayout(scrollLayout);
    scrollArea->setWidget(scrollWidget);

    // Add the scroll area to the main layout
    layout->addWidget(scrollArea);

    // Tooltips for buttons
    savePresetButton->setToolTip("Save the current settings as a preset.");
    loadPresetButton->setToolTip("Load a previously saved preset.");
    renamePresetButton->setToolTip("Rename the selected preset.");
    deletePresetButton->setToolTip("Delete the selected preset.");
    startButton->setToolTip("Start screen capturing and processing.");
    stopButton->setToolTip("Stop the screen capture.");
    previewButton->setToolTip("Preview the motion and pattern image.");
    selectRegionButton->setToolTip("Define a specific area of the screen to use.");
    addButton->setToolTip("Manually add a light's IP address and set the name.");
    removeButton->setToolTip("Remove the selected light from the list.");
    renameButton->setToolTip("Change the selected light's name.");
    resetButton->setToolTip("Reset all settings to default values.");
    discoverButton->setToolTip("Search for connected WiZ lights and populate the Light List.");

    // Tooltips for sliders
    darknessSlider->setToolTip("Adjust the darkness intensity level.");
    colorBoostSlider->setToolTip("Set the intensity of the color boost effect.");
    darknessThresholdSlider->setToolTip("Define the threshold for considering a color dark.");
    mainImageScaleSlider->setToolTip("Scale the main image for processing. Low values use less resources but impact accuracy.");
    motionImageScaleSlider->setToolTip("Scale motion images for analysis. Low values use less resources but impact accuracy.");
    patternImageScaleSlider->setToolTip("Scale pattern images for visualization. Low values use less resources but impact accuracy.");
    maxBrightnessSlider->setToolTip("Define the maximum brightness the light can reach.");

    // Tooltips for checkboxes
    brightnessCheckbox->setToolTip("Enable or disable brightness adjustment based on the darkness of the color.");
    dynamicBrightnessCheckbox->setToolTip("Allow brightness to dynamically adjust automatically.");
    colorBoostCheckbox->setToolTip("Enable the color boost effect.");
    ambilightModeCheckbox->setToolTip("Enable Ambilight mode for immersive lighting.");

    leftLightComboBox->setToolTip("Select the light to control for the left half of the screen.");
    rightLightComboBox->setToolTip("Select the light to control for the right half of the screen.");
    lightList->setToolTip("Displays the list of WiZ lights that will be used.");
    monitorComboBox->setToolTip("Select the monitor to use.");

    // Connect signals and slots
    QObject::connect(startButton, &QPushButton::clicked, StartScreenCapture);
    QObject::connect(stopButton, &QPushButton::clicked, StopScreenCapture);
    QObject::connect(previewButton, &QPushButton::clicked, StartPreview); // Corrected signal-slot connection
    QObject::connect(discoverButton, &QPushButton::clicked, DiscoverLights);
    QObject::connect(addButton, &QPushButton::clicked, &AddLight);
    QObject::connect(removeButton, &QPushButton::clicked, &RemoveLight);
    QObject::connect(renameButton, &QPushButton::clicked, RenameLight);
    QObject::connect(lightList->model(), &QAbstractItemModel::rowsInserted, [=]() {
        int itemCount = lightList->count();
        int itemHeight = lightList->sizeHintForRow(0);
        int totalHeight = itemHeight * itemCount + 2 * lightList->frameWidth();

        // Set the minimum and maximum height to be the same to make it consistent
        lightList->setMinimumHeight(totalHeight);
        lightList->setMaximumHeight(totalHeight);

        // Adjust the width based on content
        int totalWidth = std::max(lightList->sizeHintForColumn(0) + 2 * lightList->frameWidth(), baseMinimumWidth);
        lightList->setMinimumWidth(totalWidth);
        lightList->setMaximumWidth(totalWidth);
    });

    QObject::connect(lightList->model(), &QAbstractItemModel::rowsRemoved, [=]() {
        int itemCount = lightList->count();
        int itemHeight = lightList->sizeHintForRow(0);
        int totalHeight = itemHeight * itemCount + 2 * lightList->frameWidth();

        // Set the minimum and maximum height to be the same to make it consistent
        lightList->setMinimumHeight(totalHeight);
        lightList->setMaximumHeight(totalHeight);

        // Adjust the width based on content
        int totalWidth = std::max(lightList->sizeHintForColumn(0) + 2 * lightList->frameWidth(), baseMinimumWidth);
        lightList->setMinimumWidth(totalWidth);
        lightList->setMaximumWidth(totalWidth);
    });
    QObject::connect(monitorComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [](int index) {
        if (index >= 0 && index < monitors.size()) {
            selectedRegion = monitors[index].bounds;
            regionSelected = true;
            std::cout << "Selected Monitor: " << monitors[index].name << std::endl;
        }
    });

    QObject::connect(brightnessCheckbox, &QCheckBox::stateChanged, [](int state) {
        std::cout << "Adjust Brightness Checkbox state: " << state << std::endl;
        adjustBrightness = (state == Qt::Checked);
    });
    
    // Connect slider events
    QObject::connect(darknessSlider, &QSlider::valueChanged, [=](int value) {
        // Update the value label text
        darknessSliderValueLabel->setText(QString::number(value));

        // Set font and style
        QFont font = darknessSliderValueLabel->font();
        font.setPointSize(10);
        darknessSliderValueLabel->setFont(font);

        // Configure label size, alignment, and style
        darknessSliderValueLabel->setFixedSize(30, 20);
        darknessSliderValueLabel->setAlignment(Qt::AlignCenter);
        darknessSliderValueLabel->setStyleSheet(
            "background-color: black; color: white; border: 1px solid white; border-radius: 4px;"
        );

        // Calculate position
        int sliderWidth = darknessSlider->width();
        int maxRange = darknessSlider->maximum();
        int minRange = darknessSlider->minimum();
        int position = (value - minRange) * (sliderWidth - darknessSliderValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - darknessSliderValueLabel->width());

        // Move label to its position above the slider
        darknessSliderValueLabel->move(
            darknessSlider->geometry().x() + position,
            darknessSlider->geometry().y() + 25
        );

        // Update the label display
        darknessSliderValueLabel->update();
    });


    QObject::connect(darknessSlider, &QSlider::sliderPressed, [=]() {
        int value = darknessSlider->value();

        // Calculate label position
        int sliderWidth = darknessSlider->width();
        int maxRange = darknessSlider->maximum();
        int minRange = darknessSlider->minimum();
        int position = (value - minRange) * (sliderWidth - darknessSliderValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - darknessSliderValueLabel->width());

        // Move label above the slider
        darknessSliderValueLabel->move(
            darknessSlider->geometry().x() + position,
            darknessSlider->geometry().y() + 25
        );

        darknessSliderValueLabel->update();
        darknessSliderValueLabel->show();
        darknessSliderValueLabel->raise();
    });

    QObject::connect(darknessSlider, &QSlider::sliderReleased, [=]() {
        // Hide the label after a second
        QTimer::singleShot(1000, [=]() {
            darknessSliderValueLabel->hide();
        });
    });
    QObject::connect(colorBoostCheckbox, &QCheckBox::stateChanged, [](int state) {
        colorBoostEnabled = (state == Qt::Checked);
    });

    QObject::connect(colorBoostSlider, &QSlider::valueChanged, [](int value) {
        colorBoostIntensity = value;
    });
    QObject::connect(colorBoostSlider, &QSlider::valueChanged, [=](int value) {
        colorBoostValueLabel->setText(QString::number(value));

        QFont font = colorBoostValueLabel->font();
        font.setPointSize(10);
        colorBoostValueLabel->setFont(font);

        colorBoostValueLabel->setFixedSize(30, 20);
        colorBoostValueLabel->setAlignment(Qt::AlignCenter);
        colorBoostValueLabel->setStyleSheet(
            "background-color: black; color: white; border: 1px solid white; border-radius: 4px;"
        );

        int sliderWidth = colorBoostSlider->width();
        int maxRange = colorBoostSlider->maximum();
        int minRange = colorBoostSlider->minimum();
        int position = (value - minRange) * (sliderWidth - colorBoostValueLabel->width()) / (maxRange - minRange);

        position = std::clamp(position, 0, sliderWidth - colorBoostValueLabel->width());

        colorBoostValueLabel->move(
            colorBoostSlider->geometry().x() + position,
            colorBoostSlider->geometry().y() + 25
        );

        colorBoostValueLabel->update();
    });
    QObject::connect(colorBoostSlider, &QSlider::sliderPressed, [=]() {
        int value = colorBoostSlider->value();

        // Calculate label position
        int sliderWidth = colorBoostSlider->width();
        int maxRange = colorBoostSlider->maximum();
        int minRange = colorBoostSlider->minimum();
        int position = (value - minRange) * (sliderWidth - colorBoostValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - colorBoostValueLabel->width());

        // Move label above the slider
        colorBoostValueLabel->move(
            colorBoostSlider->geometry().x() + position,
            colorBoostSlider->geometry().y() + 25
        );

        colorBoostValueLabel->update();
        colorBoostValueLabel->show();
    });

    QObject::connect(colorBoostSlider, &QSlider::sliderReleased, [=]() {
        QTimer::singleShot(1000, [=]() {
            colorBoostValueLabel->hide();
        });
    });

    QObject::connect(darknessThresholdSlider, &QSlider::valueChanged, [](int value) {
        // Use the new threshold value as needed
    });
    // Connect slider events
    QObject::connect(darknessThresholdSlider, &QSlider::valueChanged, [=](int value) {
        // Update label text
        darknessThresholdValueLabel->setText(QString::number(value));

        // Set font and size
        QFont font = darknessThresholdValueLabel->font();
        font.setPointSize(10);
        darknessThresholdValueLabel->setFont(font);

        // Ensure label size and alignment
        darknessThresholdValueLabel->setFixedSize(30, 20);
        darknessThresholdValueLabel->setAlignment(Qt::AlignCenter);

        // Set style for visibility
        darknessThresholdValueLabel->setStyleSheet(
            "background-color: black; color: white; border: 1px solid white; border-radius: 4px;"
        );

        // Calculate label position
        int sliderWidth = darknessThresholdSlider->width();
        int maxRange = darknessThresholdSlider->maximum();
        int minRange = darknessThresholdSlider->minimum();
        int position = (value - minRange) * (sliderWidth - darknessThresholdValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - darknessThresholdValueLabel->width());

        // Move label above the slider
        darknessThresholdValueLabel->move(
            darknessThresholdSlider->geometry().x() + position,
            darknessThresholdSlider->geometry().y() + 25
        );

        // Update the label
        darknessThresholdValueLabel->update();

    });
    QObject::connect(darknessThresholdSlider, &QSlider::sliderPressed, [=]() {
        int value = darknessThresholdSlider->value();

        // Calculate label position
        int sliderWidth = darknessThresholdSlider->width();
        int maxRange = darknessThresholdSlider->maximum();
        int minRange = darknessThresholdSlider->minimum();
        int position = (value - minRange) * (sliderWidth - darknessThresholdValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - darknessThresholdValueLabel->width());

        // Move label above the slider
        darknessThresholdValueLabel->move(
            darknessThresholdSlider->geometry().x() + position,
            darknessThresholdSlider->geometry().y() + 25
        );

        darknessThresholdValueLabel->update();
        darknessThresholdValueLabel->show();
    });

    QObject::connect(darknessThresholdSlider, &QSlider::sliderReleased, [=]() {
        // Hide the label after a second
        QTimer::singleShot(1000, [=]() {
            darknessThresholdValueLabel->hide();
        });
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
    QObject::connect(mainImageScaleSlider, &QSlider::valueChanged, [=](int value) {
        mainImageScaleValueLabel->setText(QString::number(value));

        QFont font = mainImageScaleValueLabel->font();
        font.setPointSize(10);
        mainImageScaleValueLabel->setFont(font);

        mainImageScaleValueLabel->setFixedSize(30, 20);
        mainImageScaleValueLabel->setAlignment(Qt::AlignCenter);
        mainImageScaleValueLabel->setStyleSheet(
            "background-color: black; color: white; border: 1px solid white; border-radius: 4px;"
        );

        int sliderWidth = mainImageScaleSlider->width();
        int maxRange = mainImageScaleSlider->maximum();
        int minRange = mainImageScaleSlider->minimum();
        int position = (value - minRange) * (sliderWidth - mainImageScaleValueLabel->width()) / (maxRange - minRange);

        position = std::clamp(position, 0, sliderWidth - mainImageScaleValueLabel->width());

        mainImageScaleValueLabel->move(
            mainImageScaleSlider->geometry().x() + position,
            mainImageScaleSlider->geometry().y() + 25
        );

        mainImageScaleValueLabel->update();
    });
    QObject::connect(mainImageScaleSlider, &QSlider::sliderPressed, [=]() {
        int value = mainImageScaleSlider->value();

        // Calculate label position
        int sliderWidth = mainImageScaleSlider->width();
        int maxRange = mainImageScaleSlider->maximum();
        int minRange = mainImageScaleSlider->minimum();
        int position = (value - minRange) * (sliderWidth - mainImageScaleValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - mainImageScaleValueLabel->width());

        // Move label above the slider
        mainImageScaleValueLabel->move(
            mainImageScaleSlider->geometry().x() + position,
            mainImageScaleSlider->geometry().y() + 25
        );

        mainImageScaleValueLabel->update();
        mainImageScaleValueLabel->show();
    });

    QObject::connect(mainImageScaleSlider, &QSlider::sliderReleased, [=]() {
        QTimer::singleShot(1000, [=]() {
            mainImageScaleValueLabel->hide();
        });
    });

    QObject::connect(motionImageScaleSlider, &QSlider::valueChanged, [](int value) {
        motionImageScale = value / 100.0;
    });
    QObject::connect(motionImageScaleSlider, &QSlider::valueChanged, [=](int value) {
        motionImageScaleValueLabel->setText(QString::number(value));

        QFont font = motionImageScaleValueLabel->font();
        font.setPointSize(10);
        motionImageScaleValueLabel->setFont(font);

        motionImageScaleValueLabel->setFixedSize(30, 20);
        motionImageScaleValueLabel->setAlignment(Qt::AlignCenter);
        motionImageScaleValueLabel->setStyleSheet(
            "background-color: black; color: white; border: 1px solid white; border-radius: 4px;"
        );

        int sliderWidth = motionImageScaleSlider->width();
        int maxRange = motionImageScaleSlider->maximum();
        int minRange = motionImageScaleSlider->minimum();
        int position = (value - minRange) * (sliderWidth - motionImageScaleValueLabel->width()) / (maxRange - minRange);

        position = std::clamp(position, 0, sliderWidth - motionImageScaleValueLabel->width());

        motionImageScaleValueLabel->move(
            motionImageScaleSlider->geometry().x() + position,
            motionImageScaleSlider->geometry().y() + 25
        );

        motionImageScaleValueLabel->update();
    });
    QObject::connect(motionImageScaleSlider, &QSlider::sliderPressed, [=]() {
        int value = motionImageScaleSlider->value();

        // Calculate label position
        int sliderWidth = motionImageScaleSlider->width();
        int maxRange = motionImageScaleSlider->maximum();
        int minRange = motionImageScaleSlider->minimum();
        int position = (value - minRange) * (sliderWidth - motionImageScaleValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - motionImageScaleValueLabel->width());

        // Move label above the slider
        motionImageScaleValueLabel->move(
            motionImageScaleSlider->geometry().x() + position,
            motionImageScaleSlider->geometry().y() + 25
        );

        motionImageScaleValueLabel->update();
        motionImageScaleValueLabel->show();
    });

    QObject::connect(motionImageScaleSlider, &QSlider::sliderReleased, [=]() {
        QTimer::singleShot(1000, [=]() {
            motionImageScaleValueLabel->hide();
        });
    });

    QObject::connect(patternImageScaleSlider, &QSlider::valueChanged, [](int value) {
        patternImageScale = value / 100.0;
    });
    QObject::connect(patternImageScaleSlider, &QSlider::valueChanged, [=](int value) {
        patternImageScaleValueLabel->setText(QString::number(value));

        QFont font = patternImageScaleValueLabel->font();
        font.setPointSize(10);
        patternImageScaleValueLabel->setFont(font);

        patternImageScaleValueLabel->setFixedSize(30, 20);
        patternImageScaleValueLabel->setAlignment(Qt::AlignCenter);
        patternImageScaleValueLabel->setStyleSheet(
            "background-color: black; color: white; border: 1px solid white; border-radius: 4px;"
        );

        int sliderWidth = patternImageScaleSlider->width();
        int maxRange = patternImageScaleSlider->maximum();
        int minRange = patternImageScaleSlider->minimum();
        int position = (value - minRange) * (sliderWidth - patternImageScaleValueLabel->width()) / (maxRange - minRange);

        position = std::clamp(position, 0, sliderWidth - patternImageScaleValueLabel->width());

        patternImageScaleValueLabel->move(
            patternImageScaleSlider->geometry().x() + position,
            patternImageScaleSlider->geometry().y() + 25
        );

        patternImageScaleValueLabel->update();
    });

    QObject::connect(patternImageScaleSlider, &QSlider::sliderPressed, [=]() {
        int value = patternImageScaleSlider->value();

        // Calculate label position
        int sliderWidth = patternImageScaleSlider->width();
        int maxRange = patternImageScaleSlider->maximum();
        int minRange = patternImageScaleSlider->minimum();
        int position = (value - minRange) * (sliderWidth - patternImageScaleValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - patternImageScaleValueLabel->width());

        // Move label above the slider
        patternImageScaleValueLabel->move(
            patternImageScaleSlider->geometry().x() + position,
            patternImageScaleSlider->geometry().y() + 25
        );

        patternImageScaleValueLabel->update();
        patternImageScaleValueLabel->show();
    });

    QObject::connect(patternImageScaleSlider, &QSlider::sliderReleased, [=]() {
        QTimer::singleShot(1000, [=]() {
            patternImageScaleValueLabel->hide();
        });
    });


    QObject::connect(maxBrightnessSlider, &QSlider::valueChanged, [](int value) {
        std::cout << "MaxBrightnessSlider value: " << value << std::endl;
        maxBrightness = value;
    });
    QObject::connect(maxBrightnessSlider, &QSlider::valueChanged, [=](int value) {
        maxBrightnessValueLabel->setText(QString::number(value));

        QFont font = maxBrightnessValueLabel->font();
        font.setPointSize(10);
        maxBrightnessValueLabel->setFont(font);

        maxBrightnessValueLabel->setFixedSize(30, 20);
        maxBrightnessValueLabel->setAlignment(Qt::AlignCenter);
        maxBrightnessValueLabel->setStyleSheet(
            "background-color: black; color: white; border: 1px solid white; border-radius: 4px;"
        );

        int sliderWidth = maxBrightnessSlider->width();
        int maxRange = maxBrightnessSlider->maximum();
        int minRange = maxBrightnessSlider->minimum();
        int position = (value - minRange) * (sliderWidth - maxBrightnessValueLabel->width()) / (maxRange - minRange);

        position = std::clamp(position, 0, sliderWidth - maxBrightnessValueLabel->width());

        maxBrightnessValueLabel->move(
            maxBrightnessSlider->geometry().x() + position,
            maxBrightnessSlider->geometry().y() + 25
        );

        maxBrightnessValueLabel->update();
    });

    QObject::connect(maxBrightnessSlider, &QSlider::sliderPressed, [=]() {
        int value = maxBrightnessSlider->value();

        // Calculate label position
        int sliderWidth = maxBrightnessSlider->width();
        int maxRange = maxBrightnessSlider->maximum();
        int minRange = maxBrightnessSlider->minimum();
        int position = (value - minRange) * (sliderWidth - maxBrightnessValueLabel->width()) / (maxRange - minRange);

        // Ensure position is within bounds
        position = std::clamp(position, 0, sliderWidth - maxBrightnessValueLabel->width());

        // Move label above the slider
        maxBrightnessValueLabel->move(
            maxBrightnessSlider->geometry().x() + position,
            maxBrightnessSlider->geometry().y() + 25
        );

        maxBrightnessValueLabel->update();
        maxBrightnessValueLabel->show();
    });
    QObject::connect(maxBrightnessSlider, &QSlider::sliderReleased, [=]() {
        QTimer::singleShot(1000, [=]() {
            maxBrightnessValueLabel->hide();
        });
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
    // Connect buttons to functions
    QObject::connect(savePresetButton, &QPushButton::clicked, []() {
        bool ok;
        QString presetName = QInputDialog::getText(nullptr, "Save Preset", "Preset Name:", QLineEdit::Normal, "", &ok);
        if (ok && !presetName.isEmpty()) {
            QFile file(presetsDirectory + presetName + ".json");
            if (file.exists()) {
                QMessageBox::StandardButton reply;
                reply = QMessageBox::question(nullptr, "Overwrite Preset", "A preset with this name already exists. Do you want to overwrite it?",
                                              QMessageBox::Yes | QMessageBox::No);
                if (reply == QMessageBox::No) {
                    return;
                }
            }
            SavePreset(presetName);
        }
    });

    QObject::connect(loadPresetButton, &QPushButton::clicked, []() {
        QDir dir(presetsDirectory);
        QStringList presetFiles = dir.entryList(QStringList() << "*.json", QDir::Files);
        QList<QString> presetNames;
        for (const QString& fileName : presetFiles) {
            presetNames.append(fileName.chopped(5)); // Remove the ".json" extension
        }

        if (presetNames.isEmpty()) {
            QMessageBox::information(nullptr, "Load Preset", "No presets available to load.");
            return;
        }

        bool ok;
        QString presetName = QInputDialog::getItem(nullptr, "Load Preset", "Select Preset:", presetNames, 0, false, &ok);
        if (ok && !presetName.isEmpty()) {
            LoadPreset(presetName);
        }
    });

    QObject::connect(renamePresetButton, &QPushButton::clicked, []() {
        QDir dir(presetsDirectory);
        QStringList presetFiles = dir.entryList(QStringList() << "*.json", QDir::Files);
        QList<QString> presetNames;
        for (const QString& fileName : presetFiles) {
            presetNames.append(fileName.chopped(5)); // Remove the ".json" extension
        }

        if (presetNames.isEmpty()) {
            QMessageBox::information(nullptr, "Rename Preset", "No presets available to rename.");
            return;
        }

        bool ok;
        QString oldName = QInputDialog::getItem(nullptr, "Rename Preset", "Select Preset to Rename:", presetNames, 0, false, &ok);
        if (ok && !oldName.isEmpty()) {
            QString newName = QInputDialog::getText(nullptr, "Rename Preset", "New Preset Name:", QLineEdit::Normal, "", &ok);
            if (ok && !newName.isEmpty()) {
                QFile newFile(presetsDirectory + newName + ".json");
                if (newFile.exists()) {
                    QMessageBox::StandardButton reply;
                    reply = QMessageBox::question(nullptr, "Overwrite Preset", "A preset with this name already exists. Do you want to overwrite it?",
                                                  QMessageBox::Yes | QMessageBox::No);
                    if (reply == QMessageBox::No) {
                        return;
                    }
                }
                RenamePreset(oldName, newName);
            }
        }
    });

    QObject::connect(deletePresetButton, &QPushButton::clicked, []() {
        QDir dir(presetsDirectory);
        QStringList presetFiles = dir.entryList(QStringList() << "*.json", QDir::Files);
        QList<QString> presetNames;
        for (const QString& fileName : presetFiles) {
            presetNames.append(fileName.chopped(5)); // Remove the ".json" extension
        }

        if (presetNames.isEmpty()) {
            QMessageBox::information(nullptr, "Delete Preset", "No presets available to delete.");
            return;
        }

        bool ok;
        QString presetName = QInputDialog::getItem(nullptr, "Delete Preset", "Select Preset to Delete:", presetNames, 0, false, &ok);
        if (ok && !presetName.isEmpty()) {
            QMessageBox::StandardButton reply;
            reply = QMessageBox::question(nullptr, "Delete Preset", "Are you sure you want to delete the preset '" + presetName + "'?",
                                          QMessageBox::Yes | QMessageBox::No);
            if (reply == QMessageBox::Yes) {
                DeletePreset(presetName);
            }
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

    // Populate the monitor list
    PopulateMonitorComboBox();  // Ensure this is called here

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
