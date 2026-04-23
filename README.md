 AI-Powered Urban Mobility & Parking Assistant 🚗📍

 (Heritage & Mobility Volunteer Interface) is a Computer Vision-driven project developed to solve urban parking congestion in Madurai. By leveraging satellite imagery and YOLOv8, HemVI identifies real-time parking availability, ranks slots by proximity to the user, and provides precise GPS coordinates for navigation.

## 🌟 Key Features
- **Automated Road Band Detection**: Uses adaptive thresholding to identify tarmac boundaries without manual intervention.
- **Precision Gap Analysis**: Calculates empty spaces between vehicles down to the pixel level.
- **GPS-Pixel Mapping**: Converts image coordinates to real-world Latitude/Longitude using Web Mercator projection.
- **Smart Ranking**: Employs the Haversine formula to sort available spots by distance from the user's current location.
- **Adjacent Slot Identification**: Distinguishes between completely free road segments (Green) and prime spots next to existing vehicles (Orange).

## 🛠️ Tech Stack
- **AI/ML**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV (cv2)
- **Data Analysis**: NumPy, Math
- **Visualization**: Matplotlib
- **Platform**: Google Colab / Python 3.10+

## 📸 Visualization Legend
| Element | Color | Meaning |
| :--- | :--- | :--- |
| **Red Box** | Parked Car | Detected vehicle occupying space. |
| **Green Box** | Free Slot | Verified parking space (Size: >6.0m). |
| **Orange Box** | Adjacent Slot | Prime spot located directly next to a parked car. |
| **Yellow Line** | Road Edge | The detected boundary of the asphalt road. |

## 🚀 How It Works
1. **Inference**: The model performs high-resolution inference on a zoomed-in "slice" of the satellite image.
2. **Filtering**: Detections are filtered into 'Shoulder Zones' to ensure parking isn't suggested in the middle of the driving lane.
3. **Merging**: Overlapping detection intervals are merged to ensure safety buffers.
4. **Output**: A ranked list of the nearest GPS-tagged slots is generated for the driver.

## 📈 Roadmap
- [ ] Integration with real-time drone feeds.
- [ ] Mobile Application (Flutter/React Native) for HemVI volunteers.
- [ ] Predictive occupancy based on historical Madurai traffic data.

---
*Developed as part of the CSE 4th Semester Curriculum, 2026.
