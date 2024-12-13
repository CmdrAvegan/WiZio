# WiZ Light Screen Sync

WiZ Light Screen Sync is a C++ application that dynamically synchronizes the colors of your WiZ smart lights with the dominant colors of your computer screen. This project is designed for users who want an immersive lighting experience that matches their screen's content in real-time.

## Features

- **Dynamic Color Sync**: Automatically calculates the dominant colors of your screen and updates the WiZ lights accordingly.
- **Screen Region Selection**: Allows users to select specific regions of the screen to monitor for color changes.
- **Multi-Light Support**: Assign different WiZ lights to specific halves of the screen (left and right) or use a single light for the entire screen.
- **Brightness and Color Adjustment**: Includes options for dynamic brightness and color boost to enhance the visual effect.
- **Settings Persistence**: Save and load light configurations and user preferences to/from a settings file.
- **Customizable Controls**: Adjust brightness, color boost, and darkness threshold directly from the GUI.

## Prerequisites

- **WiZ Smart Lights**: Compatible with WiZ smart bulbs.
- **C++ Development Environment**: Requires a compiler that supports C++17 or later.
- **Libraries**:
  - [OpenCV](https://opencv.org/): For screen capture and image processing.
  - [Boost.Asio](https://www.boost.org/doc/libs/release/libs/asio/): For network communication.
  - [Qt](https://www.qt.io/): For the GUI.
  - [nlohmann/json](https://github.com/nlohmann/json): For settings management.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/CmdrAvegan/wiz-light-screen-sync.git
    cd wiz-light-screen-sync
    ```

2. Install the required libraries:
    - Follow the installation instructions for [OpenCV](https://docs.opencv.org/).
    - Install Boost libraries.
    - Install Qt and configure your environment.

3. Build the project:
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

4. Run the application:
    ```bash
    ./WiZLightScreenSync
    ```

## Usage

1. Launch the application.
2. Add your WiZ lights by entering their name and IP address.
3. (Optional) Use the **Select Region** button to define a specific area of the screen to monitor.
4. Use the dropdown menus to assign lights to the left and right halves of the screen.
5. Click **Start** to begin syncing.
6. Adjust brightness, color boost, and other settings using the provided sliders and checkboxes.

## Configuration Files

Settings are saved in a JSON file named `settings.json` in the application's directory. This file includes:
- Light names and IP addresses.
- Brightness, color boost, and darkness threshold values.

You can modify this file manually or reset settings to defaults using the **Reset to Defaults** button in the GUI.

## Screenshots

*(Add screenshots of the application here)*

## Known Issues

- **Limited Light Discovery**: Lights must be added manually by entering their IP address.
- **Single Monitor Support**: Currently optimized for single-monitor setups.
- **Performance**: May exhibit slight delays on lower-spec systems during real-time processing.

## Future Enhancements

- Automatic discovery of WiZ lights on the network.
- Enhanced multi-monitor support.
- Additional region-selection options.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas for improvements or bug fixes.

## Contact

For questions or feedback, feel free to reach out at your.email@example.com.




