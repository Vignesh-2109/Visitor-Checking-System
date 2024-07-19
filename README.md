# 🕵️‍♂️ Visitor Checking System 🚪

![Python](https://img.shields.io/badge/Python-Programming_Language-blue)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Model-orange)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-success)

## 📜 Overview

- Devised and executed a **Visitor Checking System** model with **97% accuracy** by employing advanced one-shot learning techniques.
- Utilized **HaarCascade** for face detection and **FaceNet** for converting faces into embeddings, ensuring robust verification.
- Developed a real-time authentication web application using **Flask** for seamless visitor verification.

## 👁️ Face Verification

- **Face Verification**: Implemented using **Siamese Networks** to compare face embeddings.
- **Detection & Verification**: Combined **HaarCascade** for face detection and **FaceNet** for face verification.
- **Functionalities**: Includes functions to add, delete, and verify faces efficiently.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Machine Learning**: Advanced learning techniques for model accuracy.
- **Flask**: Web framework for building the authentication web app.
- **OpenCV**: Computer vision library for face detection.

## 📂 Project Structure

```
.
├── app # Flask web application
│ ├── static # Static files (CSS, JavaScript, images)
│ ├── templates # HTML templates
│ ├── face_verification # Face verification logic
│ │ ├── haarcascade # HaarCascade for face detection
│ │ ├── facenet # FaceNet for face embedding
│ │ ├── verification.py # Face verification functions
│ ├── main.py # Entry point for the Flask app
│ └── requirements.txt # Python dependencies
├── data # Sample data and models
│ └── faces # Face images for training and verification
├── .gitignore # Git ignore file
├── README.md # Project README file
```

## 🚀 Getting Started

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/Vignesh-2109/visitor-checking-system.git
    cd visitor-checking-system
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```sh
    python main.py
    ```

4. **Access the Web App**:
    - Open your browser and go to `http://localhost:5000` to access the Visitor Checking System.

## 📚 Documentation

- [Python Documentation](https://docs.python.org/3/)
- [Machine Learning Techniques](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.



---

Developed with ❤️ by Vignesh Maram

