# Room Label Detectors

**Current**: 
1. Use contour detection or HSV color ranges to find shapes similar to room labels.
2. Use **tesseract** to detect room numbers in the room labels.
3. Use detected room label corners and room number to determine the location on the floor

**Idea for future**: Use pretrained Machine Learning model or train our own model for both text recognition and object detection, and implement in Android app using **Tensorflow Lite**.
