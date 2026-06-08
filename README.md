# Fake Plate Vehicle Detection System (Web) V1.0

## 1. Project Introduction

This project is a deep learning-based web system for detecting fake/counterfeit license plate vehicles, integrating vehicle detection, license plate localization, character recognition, vehicle type classification, color recognition, and information comparison logic to achieve intelligent vehicle license plate recognition and fake plate anomaly detection. Users can upload vehicle images through the web interface; the system automatically performs vehicle localization, license plate extraction, character recognition, vehicle type and color classification, and finally compares the results with the registered database to output recognition results and fake plate risk assessment. The project features complete frontend interaction and backend inference capabilities, independently developed from algorithm integration, model training, API development to web visualization deployment.

## 2. Tech Stack

- **Frontend**: HTML, CSS, JavaScript, implementing page layout, image upload, result display, history query, vehicle database management, and other interactive features.
- **Backend**: Python (Flask), building RESTful service interfaces, handling frontend requests, calling recognition algorithms, executing business logic and database operations.
- **Core Algorithms**:
  - YOLO-based vehicle detection algorithm for precise localization of vehicles and license plate regions;
  - HyperLPR + multi-level image enhancement strategy for license plate OCR character recognition;
  - MiniVGGNet-based vehicle type (4 classes) and color (8 colors) classification algorithm;
  - HSV color space-based intelligent color classification optimization algorithm.
- **Database**: SQLite, persistently storing vehicle registration information, recognition history, and feedback data.
- **Deployment**: Local server deployment (Docker containerization supported), achieving frontend-backend integration and real-time recognition inference.

## 3. Core Feature Modules

- **Image Upload and Preview Module**: Supports local vehicle image upload, frontend real-time preview, format and size validation.
- **Vehicle Detection and Localization Module**: Precisely locates vehicles and license plate regions in images, completes license plate correction and cropping.
- **License Plate Character Recognition Module**: Performs OCR recognition on cropped license plate regions, outputs complete license plate numbers.
- **Vehicle Type and Color Classification Module**: Intelligently classifies detected vehicles by type (sedan/bus/minivan/truck) and color (black/blue/brown/green/red/silver/white/yellow).
- **Fake Plate Detection Module**: Compares recognized license plate information with the registered database to determine if the vehicle has fake plate anomalies, outputs risk warnings.
- **Result Visualization Module**: Web interface displays original images, detection box annotations, license plate regions, recognized numbers, vehicle type and color, judgment results with clear and intuitive interaction.
- **History Record Management Module**: Supports paginated query of recognition records, statistical analysis, and persistent storage.
- **Vehicle Database Management Module**: Supports CRUD operations for vehicle registration information and CSV batch import.
- **Batch Analysis Module**: Supports simultaneous upload of multiple images for batch fake plate detection.

## 4. Runtime Environment and Usage Instructions

### Runtime Environment

- Python 3.8 and above
- Windows / macOS / Linux systems
- CPU inference supported, NVIDIA GPU acceleration recommended

### Dependency Installation

```bash
pip install -r requirements.txt
```

### Model Download (Required for First Use)

Model weight files (~200MB) are not included in the source repository and must be downloaded before first run:

**Option 1: One-click Script (Recommended)**

```bash
# Linux / macOS / Git Bash
bash download_model.sh

# Windows
python download_models.py
```

**Option 2: Manual Download**

Download the following files from [GitHub Release v1.0.0](https://github.com/WangYuning111/Find-Fake-Plate-Vehicle-web/releases/tag/v1.0.0) to the `weights/` directory:

| File Name | Description | Size |
|-----------|-------------|------|
| `best.pt` | YOLO vehicle detection model | ~15MB |
| `vehicle_type.pth` | Vehicle type classification model (4 classes) | ~80MB |
| `vehicle_color.pth` | Vehicle color classification model (8 colors) | ~80MB |
| `vehicle_brand_resnet18.pth` | Vehicle brand classification model (26 brands) | ~44MB |

> If trained model files already exist locally, they can be directly copied from the `cfg/` directory to `weights/`:
> ```bash
> cp cfg/*.pth cfg/*.pt weights/
> ```

### Startup Method

```bash
python app.py
```

Visit http://localhost:8090 to upload images for fake plate vehicle detection testing.

### Docker Deployment (Optional)

```bash
docker-compose up -d
```

### Model Notes

The algorithm inference code of this project is independently integrated and optimized. Pre-trained weight files are only used for model inference and are not included in the source code copyright registration scope. They can be downloaded or retrained during runtime.

## 5. Project Development Division (Two-Person Collaboration)

### Lead Developer: Wang Yuning (Core Development Lead)

- Responsible for overall project requirement analysis, system architecture design, and overall function planning;
- Completed backend service setup (Flask), RESTful API interface development, frontend-backend integration logic;
- Responsible for integration, code adaptation, and inference logic optimization of YOLO vehicle detection algorithm and HyperLPR license plate OCR recognition algorithm;
- Responsible for MiniVGGNet vehicle type/color classification model training strategy design, data augmentation, label smoothing, and model tuning;
- Independently wrote the core fake plate detection business logic (license plate-brand database comparison, anomaly recognition rule design);
- Responsible for optimization and accuracy improvement of HSV color space color classification algorithm;
- Responsible for overall project debugging, feature iteration, code optimization, and version integration;
- Responsible for SQLite database design, vehicle database management, history records, and feedback loop system development.

### Co-Developer: Su Puhang (Collaborative Auxiliary Development)

- Responsible for frontend page layout design, CSS styling, and JavaScript interactive feature development;
- Implemented frontend interactive modules such as image upload, preview, result echo, and history pagination;
- Responsible for vehicle image data collection, organization, and data annotation;
- Responsible for AI model training data preprocessing, dataset splitting, and training assistance;
- Assisted in project functional testing, BUG troubleshooting, adaptation optimization, and compatibility adjustments;
- Organized project materials, runtime screenshots, project documentation, and operation instructions.

## 6. Project Iteration Records

The project adopts phased iterative development, completing function updates through multiple Git commits, with complete independent development traces:

1. **Project Initialization**: Set up Flask project skeleton, configured directory structure, initialized Git repository;
2. **Frontend Page Setup**: Completed basic HTML/CSS/JS development for homepage, upload page, and result display page;
3. **Backend Interface Development**: Implemented core APIs for image upload, vehicle detection, license plate recognition;
4. **Algorithm Module Integration**: Integrated YOLO vehicle detection, HyperLPR license plate OCR, vehicle type/color classification models;
5. **Fake Plate Logic Development**: Implemented license plate-brand database comparison, anomaly judgment rules, and risk warnings;
6. **Data Management Development**: Implemented SQLite database, vehicle database CRUD, history record query;
7. **UI Beautification and Interaction Optimization**: Frontend style iteration, responsive adaptation, result visualization enhancement;
8. **Algorithm Accuracy Optimization**: HSV color threshold tuning, brand alias mapping, low confidence tolerance mechanism;
9. **Training Pipeline Construction**: Completed data augmentation, label smoothing, learning rate scheduling, and early stopping strategy;
10. **Model Management and Release**: Model weight Git separation, Release attachment publishing, one-click download script;
11. **BUG Fixes and Stability**: Fixed black vehicle misclassification, brand false positives, rotated image recognition issues, stable version release;
12. **Brand Label Cleaning (Round 1)**: Systematically reviewed samples marked as "Others" in `brand_labels.csv`, manually reviewed images and corrected 9 wrong annotations (FAW, Jinbei, Foton, Volkswagen, Great Wall, JMC, Toyota, etc.), "Others" ratio decreased from 30.4% to 26.3%;
13. **Brand Label Cleaning (Round 2)**: Continued cleaning "Others" labels, corrected 11 image brands (FAW, JMC, Changan, Zhonghua, Chery, Citroen, Hyundai, Jinbei, Volkswagen, etc.), solved extreme category imbalance (Volkswagen 124 vs Honda 11 vs BMW 2);
14. **Brand Label Cleaning (Round 3)**: Added corrections for 5 images (Honda, Volkswagen, Toyota, etc.), further expanded training data for few-sample brands, "Others" ratio decreased to ~24.3%;
15. **Brand Model Iterative Retraining**: Based on cleaned annotation data, performed multi-round iterative training on ResNet18 brand classification model (30 epochs, learning rate scheduling + early stopping strategy), enabling the model to learn the corrected label distribution;
16. **Brand Confidence Threshold Optimization**: Lowered the brand recognition output threshold in `inference.py` from 0.30 to 0.15, avoiding forcing correctable recognition results to "Unknown" due to low confidence;
17. **Fake Plate Judgment Logic Fix**: Removed the tolerance skip logic `detected_brand_conf < 0.65` in `database.py`, changed to judging suspected fake plate as long as the database registered brand and recognized brand do not match, eliminating the "the less accurate the brand → the less likely to be judged as fake" dead loop;
18. **Few-Sample Brand Governance**: For low-frequency brands such as Honda, BMW, Zhonghua with insufficient samples causing low recognition rates, continuously mined and supplemented positive sample annotations from the "Others" category, optimizing the model's discrimination ability for low-frequency brands;
19. **Brand Label Cleaning (Round 4~10) and Final Model Training**: Continued multi-round cleaning of annotation data, accumulated corrections for ~200+ images, reduced "Others" ratio from 30% to ~21%, and completed final training and test verification of brand classification model based on 295 cleaned annotation data;
20. **Brand Label Deep Cleaning and Batch Auto-Annotation**: Continued manual image review and correction for rounds 11~13 totaling 88 images, and based on current model high confidence batch auto-annotated 253 "Others" samples (confidence >= 0.80), "Others" ratio decreased to ~21.4% (from initial 63.5% to 21.4% among all 809 images), annotated samples increased to 636, training validation accuracy reached 89.7%, 10-round random test average correct rate 98.0%.

## 7. Project Display Screenshots

The following are actual system runtime screenshots, stored in the `screenshots/` directory:

### 1. Project Homepage

![Project Homepage](screenshots/01_home.jpg)

System main interface, displaying function entry, system introduction, and statistical information overview. Users can quickly access single image recognition, batch recognition, history records, and vehicle database management modules.

### 2. Image Upload Interface

![Image Upload](screenshots/02_upload.jpg)

Upload and preview effect after user selects vehicle images, supports drag-and-drop upload, displays thumbnail of the image to be detected, and provides a "Start Recognition" button to trigger the subsequent inference process.

### 3. Normal License Plate Recognition Result

![Normal Result](screenshots/03_result_normal.jpg)

Single image recognition result page, displaying detection box annotation, highlighted license plate region crop, recognized license plate number, vehicle type classification, color recognition result, and showing a green "Vehicle Information Normal" judgment prompt.

### 4. Fake Plate Judgment Result

![Fake Plate Result](screenshots/04_result_fake.jpg)

Fake plate risk judgment interface. When the recognized license plate number exists in the database with a registration record, but the recognized brand is inconsistent with the registered brand, the system outputs a red "Suspected Fake Plate" warning and provides detailed risk explanation and registered information comparison.

### 5. History Records Interface

![History Records](screenshots/06_history.jpg)

Recognition record list page, supports paginated query, filtering by license plate number/date, displaying license plate, vehicle type, color, brand, judgment result, and timestamp for each recognition, supports clicking to view details and delete records.

### 6. Vehicle Database Management Interface

![Vehicle Database](screenshots/07_database.jpg)

Registered vehicle information database management page, displaying fields such as license plate number, owner, vehicle type, color, and brand for registered vehicles, supports paginated browsing and keyword search.

### 7. Add Vehicle Interface

![Add Vehicle](screenshots/08_add_vehicle.jpg)

Vehicle registration information entry form, providing input and dropdown selection for fields such as license plate number, owner name, vehicle type, color, and brand, used to add new legal vehicle registrations to the database.

### 8. Feedback Loop Interface

![Feedback Loop](screenshots/09_feedback.jpg)

User feedback submission page. When system recognition results have deviations, users can upload the original image, select error type (license plate/vehicle type/color/brand/fake plate judgment), and fill in supplementary descriptions. Feedback data will be stored in `data/feedback_images/` for subsequent model iteration optimization.

---

## Appendix: Project Structure

```
.
├── app.py                  # Flask application main entry
├── config.py               # Configuration management
├── database.py             # SQLite database operations
├── inference.py            # Model inference wrapper
├── color_classifier.py     # HSV color classification algorithm
├── improve_accuracy.py     # Image enhancement and recognition optimization
├── feedback_db.py          # Feedback data management
├── train_all.py            # Model training script
├── download_model.sh       # One-click model weight download script
├── upload_release.py       # GitHub Release auto-upload script
├── weights/                # Model weight storage directory
├── cfg/                    # Training output model weights
├── data/                   # SQLite database
├── static/                 # CSS/JS/image resources
├── templates/              # HTML templates
└── datasets/               # Training datasets
```

## License

MIT
