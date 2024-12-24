# WiZio - WiZ Light Screen Sync

WiZio Light Screen Sync is a C++ application that dynamically synchronizes the colors of your WiZ smart lights with the dominant colors of your computer screen. This project is designed for users who want an immersive lighting experience that matches their screen's content in real-time.

## Features

- **Dynamic Color Sync**: Automatically calculates the dominant colors of your screen and updates the WiZ lights accordingly.
- **Screen Region Selection**: Allows users to select specific regions of the screen to monitor for color changes.
- **Multi-Light Support**: Assign different WiZ lights to specific halves of the screen (left and right) or use a single light for the entire screen.
- **Brightness and Color Adjustment**: Includes options for dynamic brightness and color boost to enhance the visual effect.
- **Settings Persistence**: Save and load light configurations and user preferences to/from a settings file.
- **Customizable Controls**: Adjust brightness, color boost, and darkness threshold directly from the GUI.
- **Presets**: Save, load, rename and delete custom preset settings easily.
- **Multi-Monitor Support**: Select which monitor to capture from.
## Prerequisites

- **WiZ Smart Lights**: Compatible with WiZ smart bulbs.
- **C++ Development Environment**: Requires a compiler that supports C++17 or later.
- **Libraries**:
  - [OpenCV](https://opencv.org/): For screen capture and image processing.
  - [Boost.Asio](https://www.boost.org/doc/libs/release/libs/asio/): For network communication.
  - [Qt](https://www.qt.io/): For the GUI.
  - [nlohmann/json](https://github.com/nlohmann/json): For settings management.

## Installation (Setup Installer)

1. Download the Installer

2. Follow the on-screen instructions to install

3. Launch the program 

## Installation (Building from source)

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
2. Add your WiZ lights by using the discover button or by entering their name and IP address.
3. Remove any light you do not wish to use from the list.
4. (Optional) Use the **Select Region** button to define a specific area of the screen to monitor.
5. (Optional) Use the dropdown menus to assign lights to the left and right halves of the screen.
6. Click **Start** to begin syncing.
7. Adjust brightness, color boost, and other settings using the provided sliders and checkboxes.

## Configuration Files

Settings are saved in a JSON file named `settings.json` in the application's directory. This file includes:
- Light names and IP addresses.
- Brightness, color boost, and darkness threshold values.

You can modify this file manually or reset settings to defaults using the **Reset to Defaults** button in the GUI.

## Preset Files

Presets are saved in the 'presets' directory as .json files and can be modified manually if needed.

## Known Issues

- **GUI Updates**: Slider value labels appear blank before adjusting the slider.
- **Maximum Brightness Value**: Currently does not override the maximum brightness of the light.
- **Performance**: May exhibit slight delays on lower-spec systems during real-time processing.

## Future Enhancements

- Enhanced pattern and motion analysis for improved light effects.
- Improved brightness and darkness options.
- Additional region-selection options.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas for improvements or bug fixes.

## Contact

For questions or feedback, feel free to reach out in the discussions.




