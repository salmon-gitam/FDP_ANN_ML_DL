# Course Title: Project-Based Learning in Artificial Neural Networks, Machine Learning, and Deep Learning

**Course Duration:** 3 Days

---

## **Day 1: Machine Learning Fundamentals with Employee Data**

**Project Overview:** Analyze and predict employee attributes using a dataset containing:

- First Name
- Gender
- Start Date
- Last Login Time
- Salary
- Bonus %
- Is Senior Management
- Team Name

### **Session 1: Data Manipulation with Pandas and NumPy**

- **Learning Objectives:**
  - Understand data structures in Pandas and NumPy.
  - Perform data loading, exploration, and manipulation.
- **Activities:**
  - **Data Loading:** Import the employee dataset into a Pandas DataFrame.
  - **Exploratory Data Analysis (EDA):** Use descriptive statistics to understand data distributions.
  - **Data Manipulation:** Filter, sort, and group data using Pandas.
  - **Numerical Computations:** Perform calculations on Salary and Bonus % using NumPy.
- **Resources:**
  - Jupyter Notebook or Google Colab.
  - Employee dataset (provided in CSV format).
- **Outcome:** Ability to manipulate and analyze data using Pandas and NumPy.

### **Session 2: Data Visualization with Matplotlib and Introduction to OpenCV**

- **Learning Objectives:**
  - Create various plots to visualize data distributions.
  - Get introduced to image processing with OpenCV.
- **Activities:**
  - **Matplotlib Visualization:**
    - Plot histograms of Salary distributions by Gender.
    - Create bar charts showing average Bonus % by Team.
    - Develop scatter plots for Salary vs. Years of Service.
- **Outcome:** Proficiency in visualizing data to uncover insights.

### **Session 3: Data Preprocessing and Basic Machine Learning Models**

- **Learning Objectives:**
  - Handle missing data and normalize datasets.
  - Understand and implement Linear Regression, Logistic Regression, and Decision Trees.
- **Activities:**
  - **Data Cleaning:**
    - Identify missing values and decide on imputation strategies.
  - **Data Normalization:**
    - Apply scaling techniques to Salary and Bonus %.
  - **Model Building:**
    - **Linear Regression:** Predict Salary based on features like Years of Service.
    - **Logistic Regression:** Classify employees as Senior Management or not.
    - **Decision Tree:** Predict the likelihood of an employee belonging to a specific team.
- **Outcome:** Ability to preprocess data and build basic predictive models.

### **Session 4: Advanced Machine Learning Techniques**

- **Learning Objectives:**
  - Implement ensemble methods and clustering algorithms.
  - Understand model evaluation techniques and regularization.
- **Activities:**
  - **Random Forest & XGBoost:**
    - Improve model accuracy using ensemble methods.
  - **K-Nearest Neighbors (KNN):**
    - Classify employees based on similar attributes.
  - **K-Means Clustering:**
    - Group employees into clusters based on Salary and Bonus %.
  - **Model Evaluation:**
    - Apply K-fold cross-validation.
  - **Regularization:**
    - Implement L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.
- **Outcome:** Mastery of advanced ML algorithms and evaluation methods.

---

## **Day 2: Deep Learning with Handwritten Digit Recognition**

**Project Overview:** Build and train neural networks to recognize handwritten digits using the MNIST dataset.

### **Session 1: Data Preprocessing and Neural Network Construction**

- **Learning Objectives:**
  - Prepare image data for neural network input.
  - Understand the architecture of feedforward neural networks.
- **Activities:**
  - **OpenCV Introduction (Optional for image-related data):**
    - Basic operations like reading and displaying images.
  - **Data Loading:**
    - Import the MNIST dataset.
  - **Data Normalization:**
    - Scale pixel values to range [0, 1].
  - **Neural Network Building:**
    - Design a simple feedforward neural network using TensorFlow/Keras or PyTorch.
- **Outcome:** Ability to construct and understand a basic neural network model.

### **Session 2: Model Training, Evaluation, and Testing**

- **Learning Objectives:**
  - Train neural networks and interpret performance metrics.
  - Test the model on unseen data.
- **Activities:**
  - **Model Training:**
    - Train the neural network on training data.
  - **Performance Evaluation:**
    - Use accuracy, precision, recall, and confusion matrix.
    - Plot training vs. validation loss and accuracy.
  - **Model Testing:**
    - Evaluate the model on the test dataset.
- **Outcome:** Proficiency in training neural networks and evaluating their performance.

### **Session 3: Building Efficient Data Pipelines**

- **Learning Objectives:**
  - Automate data loading and preprocessing steps.
  - Understand the importance of efficient data handling in deep learning.
- **Activities:**
  - **Data Pipeline Creation:**
    - Use TensorFlow's `tf.data` or PyTorch's `DataLoader`.
  - **Data Augmentation:**
    - Apply techniques like rotation, scaling, and flipping to increase dataset size.
- **Outcome:** Ability to create scalable data pipelines for large datasets.

### **Session 4: Convolutional Neural Networks (CNN) and Max Pooling**

- **Learning Objectives:**
  - Understand the role of convolutional layers and pooling in CNNs.
  - Build and train a CNN for image classification.
- **Activities:**
  - **CNN Architecture:**
    - Add convolutional and pooling layers to the existing model.
  - **Model Training:**
    - Train the CNN and compare its performance with the feedforward network.
  - **Visualization:**
    - Visualize feature maps and understand what the CNN is learning.
- **Outcome:** Competence in building and training CNNs for complex image recognition tasks.

---

## **Day 3: Advanced NLP and Computer Vision Projects**

### **Project 1: SMS Spam Classification**

**Session 1: Natural Language Processing with NLTK**

- **Learning Objectives:**
  - Perform text preprocessing techniques essential for NLP tasks.
- **Activities:**
  - **Data Loading:**
    - Import SMS Spam Collection dataset.
  - **Text Preprocessing:**
    - Tokenization of SMS messages.
    - Stemming and lemmatization to reduce words to their root forms.
    - Named Entity Recognition (NER) to identify entities.
- **Outcome:** Ability to preprocess and prepare text data for modeling.

**Session 2: Feature Extraction and Naive Bayes Classification**

- **Learning Objectives:**
  - Convert text data into numerical vectors.
  - Implement a Naive Bayes classifier for text data.
- **Activities:**
  - **Feature Extraction:**
    - Use Word2Vec embeddings with Gensim.
  - **Model Building:**
    - Train a Naive Bayes classifier to distinguish between spam and ham messages.
  - **Model Evaluation:**
    - Evaluate model performance using accuracy, precision, and recall.
- **Outcome:** Proficiency in building and evaluating text classification models.

### **Project 2: Face Recognition Using Image Embeddings**

**Session 3: Face Recognition Techniques**

- **Learning Objectives:**
  - Understand how image embeddings can be used for face recognition.
- **Activities:**
  - **Data Preparation:**
    - Collect or use a provided set of face images.
  - **Embedding Extraction:**
    - Use pre-trained models (like FaceNet) to extract embeddings.
  - **Face Recognition:**
    - Compare embeddings to identify or verify individuals.
  - **Application Development:**
    - Build a simple face recognition application.
- **Outcome:** Ability to implement face recognition systems using image embeddings.

### **Project 3: Object Detection with YOLOv8**

**Session 4: Advanced Object Detection and Counting**

- **Learning Objectives:**
  - Understand the YOLO (You Only Look Once) object detection framework.
  - Apply object detection models to real-world problems.
- **Activities:**
  - **Environment Setup:**
    - Install YOLOv8 and its dependencies.
  - **Model Execution:**
    - Run YOLOv8 on sample images and videos.
  - **Custom Object Detection:**
    - Fine-tune the model for custom objects.
  - **Object Counting:**
    - Implement counting logic for detected objects.
- **Outcome:** Competence in using state-of-the-art object detection models for practical applications.

---

# **Final Notes and Expected Competencies**

By the end of this 3-day course, participants will have:

- **Practical Experience:** Hands-on experience with real-world datasets in employee analytics, image recognition, and natural language processing.
- **Technical Skills:** Proficiency in data manipulation, visualization, machine learning algorithms, deep learning architectures, and advanced AI frameworks.
- **Project Portfolio:** Completed multiple projects that can be added to their professional portfolio.
- **Foundational Knowledge:** A solid understanding of both theoretical concepts and practical implementations in ANN, ML, and DL.

# **Prerequisites**

- Basic programming knowledge in Python.
- Familiarity with fundamental statistics and linear algebra.
- Prior exposure to machine learning concepts is helpful but not required.

# **Required Software and Tools**

- Python 3.x
- Jupyter Notebook or Google Colab
- Libraries: Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras or PyTorch, NLTK, Gensim, OpenCV, YOLOv8
