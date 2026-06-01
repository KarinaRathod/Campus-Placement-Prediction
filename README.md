# 🎓 Campus Placement & Salary Intelligence System

An AI-powered Career Analytics Platform that predicts student placement outcomes and estimated salary packages using Machine Learning and Explainable AI (XAI).

This system helps students, training departments, and career counselors make data-driven decisions by identifying the factors that most influence placement success and compensation potential.

---

## 📌 Overview

The Campus Placement & Salary Intelligence System is an end-to-end Machine Learning application designed to analyze academic performance, technical competencies, and professional experience to forecast:

* ✅ Placement Probability
* 💰 Expected Salary Package (LPA)
* 📊 Key Success Factors through Explainable AI

Unlike traditional prediction systems that only provide outcomes, this platform explains *why* a prediction was made, enabling users to understand and improve their employability profile.

---

## ✨ Key Features

### 📈 Interactive Analytics Dashboard

* Real-time exploratory data analysis (EDA)
* Placement trends visualization
* Academic performance vs placement success insights
* Salary distribution analysis
* Interactive charts powered by Plotly

### 🤖 Dual Machine Learning Architecture

#### 1. Placement Prediction Model

Uses a **Random Forest Classifier** to estimate the probability of getting placed.

**Output:**

* Placement Status Prediction
* Placement Probability Score
* Confidence Metrics

#### 2. Salary Prediction Model

Uses a **Random Forest Regressor** to estimate the expected salary package for candidates likely to be placed.

**Output:**

* Predicted Salary (LPA)
* Compensation Range Estimation

---

### 🧠 Explainable AI (XAI)

The platform provides transparent insights into model decisions by displaying:

* Feature Importance Rankings
* Career Success Drivers
* Impact Analysis of Academic Scores
* Technical Skill Contribution
* Internship & Work Experience Influence

This helps students understand which areas require improvement to increase placement chances.

---

### 🎯 Dynamic User Interface

* Automatic dataset schema detection
* Dynamic input form generation
* Intelligent categorical encoding
* User-friendly prediction workflow
* Real-time model inference

---

### ⚡ Production-Ready Design

* Data caching for faster performance
* Robust exception handling
* Modular architecture
* Scalable machine learning pipeline
* Clean and responsive UI

---

# 🏗️ System Architecture

```text
Student Data Input
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
 ┌───────────────────┐
 │ Random Forest     │
 │ Classifier        │
 └───────────────────┘
        │
        ▼
Placement Probability
        │
        ▼
If Candidate Likely Placed
        │
        ▼
 ┌───────────────────┐
 │ Random Forest     │
 │ Regressor         │
 └───────────────────┘
        │
        ▼
Predicted Salary (LPA)
        │
        ▼
Explainable AI Analysis
```

---

# 🛠️ Technology Stack

| Category             | Technology             |
| -------------------- | ---------------------- |
| Frontend             | Streamlit              |
| Data Processing      | Pandas, NumPy          |
| Visualization        | Plotly, Seaborn        |
| Machine Learning     | Scikit-Learn           |
| Classification Model | RandomForestClassifier |
| Regression Model     | RandomForestRegressor  |
| Encoding             | LabelEncoder           |
| Programming Language | Python                 |

---

# 📂 Dataset Structure

The system is optimized for a comprehensive placement dataset containing academic, technical, and professional attributes.

| Feature                | Description                       |
| ---------------------- | --------------------------------- |
| ssc_percentage         | 10th Grade Percentage             |
| hsc_percentage         | 12th Grade Percentage             |
| degree_percentage      | Undergraduate Percentage          |
| work_experience        | Previous Work Experience (Months) |
| technical_skills_score | Technical Assessment Score        |
| internships            | Number of Internships             |
| projects               | Academic/Industry Projects        |
| specialization         | Domain of Study                   |
| placed                 | Placement Status (Target)         |
| salary_lpa             | Salary Package in LPA (Target)    |

### Target Variables

#### Classification Target

```python
placed
0 = Not Placed
1 = Placed
```

#### Regression Target

```python
salary_lpa
```

---

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/placement-prediction-system.git
cd placement-prediction-system
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If no requirements file exists:

```bash
pip install streamlit pandas numpy scikit-learn plotly seaborn
```

---

## 4️⃣ Prepare Dataset

Place your dataset file in the project root directory:

```text
campus_placement_data.csv
```

---

## 5️⃣ Launch Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🔄 Prediction Workflow

### Step 1

Student enters academic and skill-related information.

### Step 2

Random Forest Classifier predicts placement probability.

### Step 3

If placement probability exceeds the predefined threshold, the salary prediction model is activated.

### Step 4

Random Forest Regressor estimates expected salary package.

### Step 5

Explainable AI module identifies the most influential features affecting the prediction.

---

# 📊 Sample Insights Generated

✔ Technical Skills Contribution

✔ Academic Performance Impact

✔ Internship Effectiveness

✔ Work Experience Influence

✔ Specialization-Based Placement Trends

✔ Salary Distribution Analysis

---

# 🎯 Use Cases

### For Students

* Assess placement readiness
* Identify skill gaps
* Understand salary expectations

### For Career Counselors

* Guide students effectively
* Track employability indicators
* Improve training strategies

### For Placement Cells

* Analyze placement trends
* Generate institutional insights
* Enhance placement outcomes

### For Educational Institutions

* Data-driven curriculum planning
* Employability benchmarking
* Performance analytics

---

# 📈 Future Enhancements

* Deep Learning-based prediction models
* Resume Analysis using NLP
* Interview Performance Assessment
* Skill Recommendation Engine
* Industry Demand Forecasting
* SHAP-based Explainability
* Real-time Job Matching System
* Multi-University Analytics Dashboard

---

# 🤝 Contributing

Contributions are welcome and greatly appreciated.

### Contribution Steps

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/AmazingFeature
```

3. Commit your changes

```bash
git commit -m "Add Amazing Feature"
```

4. Push to GitHub

```bash
git push origin feature/AmazingFeature
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed with ❤️ to empower students, placement cells, and career counselors through data-driven career intelligence.

**Campus Placement & Salary Intelligence System**
*Predict. Analyze. Improve. Succeed.*

