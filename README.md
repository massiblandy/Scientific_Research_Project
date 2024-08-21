# 🤖 Using convolutional neural networks for recognition and localization of visual landmarks for a humanoid robot
---
## Description
This repository presents the Undergraduate Scientific Research project developed to enhance the localization and decision-making capabilities of the humanoid robot from the RoboFEI team, which participates in humanoid robot soccer games. To improve the robot's efficiency in locating itself on the soccer field, an algorithm was implemented to detect landmarks on the field, as well as to measure the distance between the robot and the identified elements.
The project involved the study and application of computer vision and image recognition techniques, such as Convolutional Neural Networks. The results achieved confirm the effectiveness of the algorithm, demonstrating that the detection of landmarks and distance calculation were carried out with good precision.

---
## Objectives
The principal objectives of this project are to enhance the localization capabilities of a humanoid robot by implementing an algorithm for accurate landmark detection on the soccer field, to improve the robot's decision-making by enabling strategic actions based on its position and proximity to landmarks, and to ensure robust performance across various lighting conditions. Additionally, the project aims to lay a foundation for future developments by providing a reliable base for further enhancements in the robot's capabilities.

---
## Prerequisites
- Linux
- https://git-scm.com/

    Operating System:
        Ubuntu 20.04 or later (recommended for compatibility with ROS2)

    Python:
        Python 3.8 or later

    ROS2:
        ROS2 Humble Hawksbill (or the version you're using)
        Properly configured ROS2 workspace

    Dependencies:
        Install required Python libraries:

        bash

    pip install -r requirements.txt

    You might need additional ROS2 packages, such as:
        ros-<distro>-cv-bridge
        ros-<distro>-image-transport
        ros-<distro>-vision-msgs

CUDA (Optional):

    For GPU acceleration, install CUDA Toolkit and cuDNN if you're using a GPU.
    Ensure the correct version is installed for your PyTorch setup.

Camera Setup:

    A compatible camera (e.g., Logitech C920) configured to work with OpenCV.

YOLOv7 Weights:

    Pre-trained YOLOv7 weights (best_localization.pt) placed in the correct directory:

    bash

    mkdir -p src/vision_yolov7/vision_yolov7/peso_tiny/
    cp path_to_your_weights/best_localization.pt src/vision_yolov7/vision_yolov7/peso_tiny/

Git:

    Git for version control, if users need to clone the repository:

    bash

    sudo apt-get install git

CMake and Colcon:

    Ensure you have CMake and Colcon installed for building the ROS2 packages:

    bash

sudo apt-get install cmake
sudo apt-get install python3-colcon-common-extensions




---
## Acknowledgements
This project was developed as part of an Undergraduate Scientific Research program at Centro Universitário FEI. Special thanks to the RoboFEI team for their support and collaboration throughout the project.


This repository contains the code written using ROS2, by RoboFEI team, which is currently used in our physical robots. 

It is divided in 8 packages: 

* control: contains the code related to the robots motion and its parameters;
* custom_interfaces: contains all the custom interfaces used in the code;
* decision: contains the code responsible for the robots decision;
* GC: contains the code responsible for the robots communication with the game controller;
* localization_pkg: contains the code responsible for the robots localization;
* robotis_ws: contains the code responsible for the robots communication with its motors;
* start: contains the launch file to run all the nodes at once;
* um7: contains the code responsible for getting IMU measurements;
* vision_pkg: contains the code responsible for the robots vision.

## Installation:
1. First, download this repo from github:

    ```$ git clone https://github.com/RoboFEI/RoboFEI-HT_2023_SOFTWARE.git```

2. Then, install ROS2 Humble and all the libraries that are used in our code:

    ```$ ./comandos.sh```

3. Compile all the packages, in the source folder (*if there are more folders besides src delete them*):

    ```$ colcon build --symlink-install```

4. Setup the environment:

    ```$ source install/setup.bash```

5. Run all codes at once:

    ```$ ros2 launch start start.launch.py```

6. Run the codes separately:

    - Control: 
    
        ```$ ros2 launch control action.launch.py```

    - Decision: 
    
        ```$ ros2 run decision_pkg decision_node```

    - GC: 
    
        ```$ ros2 run controller talker```

    - IMU: 
    
        ```$ ros2 run um7 um7_node```

    - Localization: 
    
        ```$ ros2 run localization_pkg localization_node --mcl -l -g```

    - Motors: 
    
        ```$ ros2 run dynamixel_sdk_examples read_write_node```
    
    - Vision: 
    
        ```$ ros2 run vision_pkg vision --vb```
