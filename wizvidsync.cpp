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
boost::asio::io_context ioContext;
udp::socket* udpSocket;
std::atomic<bool> adjustBrightness(false);  // Flag to enable/disable brightness adjustment
std::atomic<int> darknessThreshold(30);     // Threshold for darkness intensity adjustment
std::atomic<bool> colorBoostEnabled(false);  // Flag to enable/disable color boost
std::atomic<int> colorBoostIntensity(20);   // Intensity of the color boost
RECT selectedRegion;
bool selectingRegion = false;
bool regionSelected = false;
// Global timer for screen capture
QTimer* captureTimer = nullptr;


std::vector<Light> lights;  // Vector to store the lights

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
    cv::resize(mat, mat, cv::Size(), 0.5, 0.5); // Scale down the image to 25% of the original size

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

void SendColorToWiZ(cv::Scalar color, double averageBrightness, const QString& lightIp) {
    if (colorBoostEnabled) {
        // Apply color boost
        color = boostSaturation(color, colorBoostIntensity);
    }

    int red = static_cast<int>(color[2]);
    int green = static_cast<int>(color[1]);
    int blue = static_cast<int>(color[0]);

    int brightness = 255;  // Default brightness
    int darknessThresholdValue = darknessThresholdSlider->value();  // Get the user-defined threshold value

    if (dynamicBrightnessCheckbox->isChecked()) {
        // Calculate the brightness value based on average brightness
        brightness = std::clamp(static_cast<int>((averageBrightness / 255.0) * 100), 1, 100);
    }

    if (red == 0 && green == 0 && blue == 0) {
        // Set brightness to 0 and color to (1, 1, 1) if the color is black
        red = green = blue = 1;
        brightness = 0;
        lightIsOff = true;
    } else {
        // Adjust brightness based on how close the color is to black
        if (adjustBrightness) {
            int maxColorValue = std::max({red, green, blue});
            if (maxColorValue < darknessThresholdValue) { // Use the user-defined threshold value
                int colorBrightness = static_cast<int>(0.299 * red + 0.587 * green + 0.114 * blue);
                brightness = std::max(1, 100 - (darknessThreshold.load() * (255 - colorBrightness) / 255));
            }
        }
        lightIsOff = false;
    }

    QString json = QString("{\"method\":\"setPilot\",\"params\":{\"r\":%1,\"g\":%2,\"b\":%3,\"dimming\":%4}}")
        .arg(red)
        .arg(green)
        .arg(blue)
        .arg(brightness);

    udp::endpoint endpoint(boost::asio::ip::make_address(lightIp.toStdString()), 38899);  // Default WiZ UDP port
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


// Start the screen capture in a separate thread
void StartScreenCapture() {
    if (!capturing) {
        capturing = true;
        std::thread([]() {
            while (capturing) {
                try {
                    // Capture the screen
                    HBITMAP hBitmap = CaptureScreen();

                    // Convert HBITMAP to cv::Mat
                    cv::Mat img = HBITMAPToMat(hBitmap);

                    // Split the image into left and right halves
                    cv::Mat leftHalf = img(cv::Rect(0, 0, img.cols / 2, img.rows));
                    cv::Mat rightHalf = img(cv::Rect(img.cols / 2, 0, img.cols / 2, img.rows));

                    // Calculate the dominant color and average brightness for each half
                    cv::Scalar leftColor = getDominantColor(leftHalf);
                    cv::Scalar rightColor = getDominantColor(rightHalf);
                    cv::Scalar leftAvgBrightness = getAverageColorBrightness(leftHalf);
                    cv::Scalar rightAvgBrightness = getAverageColorBrightness(rightHalf);

                    // Calculate average brightness
                    double leftAverageBrightness = (leftAvgBrightness[0] + leftAvgBrightness[1] + leftAvgBrightness[2]) / 3.0;
                    double rightAverageBrightness = (rightAvgBrightness[0] + rightAvgBrightness[1] + rightAvgBrightness[2]) / 3.0;

                    if (leftLightComboBox->currentIndex() == 0 && rightLightComboBox->currentIndex() == 0) {
                        // Both set to "None", use all lights
                        for (const Light& light : lights) {
                            SendColorToWiZ(leftColor, leftAverageBrightness, light.ipAddress);
                        }
                    } else {
                        // Send colors and brightness to respective lights
                        if (leftLightComboBox->currentIndex() > 0) { // > 0 to exclude "None"
                            QString leftLightIp = lights[leftLightComboBox->currentIndex() - 1].ipAddress; // -1 to account for "None"
                            SendColorToWiZ(leftColor, leftAverageBrightness, leftLightIp);
                        }
                        if (rightLightComboBox->currentIndex() > 0) { // > 0 to exclude "None"
                            QString rightLightIp = lights[rightLightComboBox->currentIndex() - 1].ipAddress; // -1 to account for "None"
                            SendColorToWiZ(rightColor, rightAverageBrightness, rightLightIp);
                        }
                    }

                    // Update the color preview box in the GUI
                    UpdateColorDisplay(leftColor);

                    DeleteObject(hBitmap);

                    // Sleep for a short interval to reduce CPU usage
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                } catch (const std::exception& ex) {
                    std::cerr << "Exception in screen capture process: " << ex.what() << std::endl;
                }
            }
        }).detach();
    }
}




// Stop the screen capture
void StopScreenCapture() {
    capturing = false;
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

    // Save whether darkness adjustment and color boost are enabled
    settings["adjustBrightness"] = adjustBrightness.load();
    settings["colorBoostEnabled"] = colorBoostEnabled.load();

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

        // Load whether darkness adjustment and color boost are enabled
        adjustBrightness = settings.value("adjustBrightness", DEFAULT_ADJUST_BRIGHTNESS);
        colorBoostEnabled = settings.value("colorBoostEnabled", DEFAULT_COLOR_BOOST_ENABLED);

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

        std::cout << "Settings loaded successfully." << std::endl;
    } else {
        std::cerr << "Could not open settings file. Creating a default settings file." << std::endl;
        CreateDefaultSettingsFile();
    }
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

    // Add color display
    colorLabel = new QLabel("Dominant Color:");
    colorDisplay = new QLabel;
    colorDisplay->setFixedSize(100, 50);
    colorDisplay->setStyleSheet("background-color: rgb(0, 0, 0);");  // Initial color
    layout->addWidget(colorLabel);
    layout->addWidget(colorDisplay);

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

    // Add color boost controls
    colorBoostCheckbox = new QCheckBox("Enable Color Boost");
    layout->addWidget(colorBoostCheckbox);
    colorBoostSlider = new QSlider(Qt::Horizontal);
    colorBoostSlider->setRange(0, 100);
    layout->addWidget(new QLabel("Color Boost Intensity"));
    layout->addWidget(colorBoostSlider);

    // Add dynamic brightness checkbox
    dynamicBrightnessCheckbox = new QCheckBox("Enable Dynamic Brightness");
    layout->addWidget(dynamicBrightnessCheckbox);

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

    // Add reset to defaults button
    QPushButton* resetButton = new QPushButton("Reset to Defaults");
    layout->addWidget(resetButton);

    // Connect signals and slots
    QObject::connect(startButton, &QPushButton::clicked, StartScreenCapture);
    QObject::connect(stopButton, &QPushButton::clicked, StopScreenCapture);
    QObject::connect(addButton, &QPushButton::clicked, &AddLight);
    QObject::connect(removeButton, &QPushButton::clicked, &RemoveLight);
    QObject::connect(brightnessCheckbox, &QCheckBox::stateChanged, [](int state) {
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
    QObject::connect(selectRegionButton, &QPushButton::clicked, []() {
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

    return result;
}
