# Indoor Localization Using Room Label Detectors

**Functions**: 
- **Room Label Detector**: detect room labels in the input images, return locations of corners and detected room names. 
    1. Use contour detection or HSV color ranges to find shapes similar to room labels.
    2. Use **tesseract** to detect room numbers in the room labels.
    - **TODO**: improve performance when light condition is not good. 

- **TODO: Localization**: use detected room label corners and room number to determine the relative location of camera with regard to the label. Then determine the location on the floor.

- **Draw Map**: A GUI tool for developers to create and modify the floorplan.
    - **Add/Remove Walls**: Define navigable and restricted areas.
    - **Set Destinations and Waypoints**: Place destinations with orientation, aiding in directional guidance.
    - **Set Room Labels**: Define size of room labels and place them in floor with orientation. 
    - This tool generates JSON-based map files, used by the pathfinding component.

**Potential alternatives**: 
- Use pretrained Machine Learning model or train our own model for both text recognition and object detection, and implement in Android app using **Tensorflow Lite**.
