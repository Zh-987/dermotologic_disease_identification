# dermotologic_disease_identification
This repository includes code for data preprocessing, model training, evaluation, and web application development. It serves as a comprehensive resource for anyone interested in dermatological disease identification using AI and machine learning techniques.

# Project Overview
This project focuses on creating an efficient system for identifying dermatological diseases through the application of machine learning and deep learning algorithms. The primary goals are: <br>
(FYI. The results of all the pictures are in the Kazakh language. This is because this scientific work was started in the Kazakh language but you can contribute it in any language.)
<li>Studying and implementing algorithms for disease classification. </li>
<li>Developing a web-based application for image-based disease diagnosis. </li>
<li>Providing insights and predictions in an intuitive user interface. </li>


## Key Features

### 1. Prediction Module
<li>Users can upload an image of the affected skin area. </li>
<li>The system uses a trained deep learning model to predict the type of dermatological disease. </li>
<li>Results are displayed with classification details and confidence scores. </li>

### 2. Dataset Management
<li>View, add, or delete records from the dataset. </li>
<li>Understand the dataset used for training and evaluation. </li>

### 3. Training Module
<li>Train the model using the MobileNetV2 architecture. </li>
<li>Enhance model performance with new data or updated algorithms. </li>

### 4. About Section
<li> Learn about the application and its underlying technologies. </li>

## Algorithm Comparison
To identify the most efficient algorithm for dermatological disease classification, three popular deep learning architectures were compared:

### 1. ResNet50
#### <li> Strengths: </li>
<li>Deeper architecture with skip connections, addressing vanishing gradient issues. </li>
<li>Good performance on large datasets. </li>

#### <li> Weaknesses:  </li>
<li>High computational cost and slower inference time.</li>
<li>Requires significant resources for training and deployment.</li>

<img src="https://github.com/user-attachments/assets/0424887d-0f3a-4fa4-8fd2-aef4acf92db3" alt="CNN ResNet 50 algorithm results" width="300" height="350">

### 2. VGG16
#### <li> Strengths:</li>
<li>Simple architecture, easy to implement and understand.</li>
<li>Performs well for small to medium datasets.</li>

#### <li> Weaknesses:</li>
<li>Large model size, leading to slower inference.</li>
<li>High memory and storage requirements.</li>

<img src="https://github.com/user-attachments/assets/cf9c5aa4-a147-410b-9300-99bba2f90100" alt="CNN VGG16 algorithm results" width="300" height="350">

### 3. MobileNetV2 (Chosen Algorithm)
#### <li> Strengths: </li>
<li>Lightweight architecture optimized for mobile and web applications.</li>
<li>Faster inference time, making it suitable for real-time predictions.</li>
<li>Comparable accuracy to ResNet50 and VGG16 with significantly fewer parameters.</li>

#### <li> Weaknesses:</li>
<li>May slightly lag in accuracy when compared to ResNet50 on highly complex datasets.</li>

<img src="https://github.com/user-attachments/assets/90cc0b5c-55f9-49cd-b09a-5f3f32ed3071" alt="CNN MobileNetV2 algorithm results" width="300" height="350">

#### <li> Comparison Results</li>

##### Comparison of the models
<img src="https://github.com/user-attachments/assets/3b4ee406-c774-4c00-aab9-47b18c818aa9" alt="Comparison of the models" width="300" height="300">

##### <li> Accuracy test of the models</li>
<img src="https://github.com/user-attachments/assets/aa36c93a-8e4d-44a1-b4ad-ea62462de663" alt="Accuracy test of the models" width="300" height="300">

##### <li> Time comparison of the models</li>
<img src="https://github.com/user-attachments/assets/ef1686bf-4516-4451-9109-6abf6053ce09" alt="Time comparison of the models" width="300" height="300">

## Architecture
The application is designed with a user-friendly interface and clear workflows. Below is an overview of its main components:

### Web Application Flow
#### 1. Home Page: Entry point for the user.
<img src="https://github.com/user-attachments/assets/08cdd09b-b2af-456b-9c86-2b5bc6d2ae11" alt="Web App" width="550" height="300">

#### 2. Prediction Workflow:
<li> Upload an image. </li>
<li> Use the trained model to predict. </li>
<li> View the prediction result. </li>
<img src="https://github.com/user-attachments/assets/e86892fa-e04b-43fc-ba1f-d4d8f51b5ab7" alt="Web APP results" width="600" height="300">

#### 3. Dataset Management:
<li>Explore the dataset. </li>
<li>Add or remove records. </li>

#### 4. Training Workflow:
<li>Train the model using the MobileNetV2 algorithm.</li>
<li>Update the system with the improved model.</li>

#### 5. About the Application:
<li>Learn more about the project's purpose and functionality.</li>
<li>Refer to the architecture diagram for a detailed visual representation of the application's structure and flow.</li>

## Technology Stack
<li>Frontend: HTML, CSS, JavaScript. </li>
<li>Backend: Python (Flask) for handling model predictions and dataset management. </li>
<li>Machine Learning: TensorFlow/Keras for model training (MobileNetV2). </li>

## Installation
1. Clone the repository: </br>
<code>git clone https://github.com/your-username/dermatologic_disease_identification.git  
cd dermatologic_disease_identification </code>

2. Set up the environment: </br>
<code>python -m venv env  
source env/bin/activate  # For Linux/Mac  
env\Scripts\activate  # For Windows  
pip install -r requirements.txt </code>

3. Start the application: </br>
<code> python app.py </code>

4. Open the application in your browser:
<code> http://localhost:5000 </code>

## Usage
<li>Prediction: Upload an image to get a disease classification. </li>
<li>Dataset Management: Add, view, or delete dataset entries to improve the model. </li>
<li>Training: Train the model with new or updated datasets. </li>

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (feature/your-feature).
3. Commit your changes.
4. Push to the branch.
5. Create a Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Let me know if you'd like any specific adjustments or if you'd like me to include more technical details!

#### Contact me: zhasulanasainov@gmail.com
