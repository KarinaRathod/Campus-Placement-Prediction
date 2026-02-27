🎓 Campus Placement & Salary Intelligence System
An end-to-end Machine Learning platform that leverages Random Forest Ensembles to predict student placement outcomes and potential salary packages (LPA). Built with a focus on Explainable AI (XAI), this system doesn't just give a "Yes" or "No"—it explains the "Why" through feature importance analytics.

🚀 Key Features
•	Interactive EDA Dashboard: Real-time data visualization of academic performance vs. placement success using Plotly.
•	Dual-Model Architecture:
•	Classification: Predicts the probability of placement using a Random Forest Classifier.
•	Regression: Estimates the expected salary for placed candidates using a Random Forest Regressor.
•	Explainable AI (XAI): Visualizes the "Key Drivers of Success" so students know which skills (e.g., Technical Score vs. Internships) impact their career most.
•	Dynamic UI: Automatically handles categorical encoding and generates input fields based on the dataset schema.
•	Production-Ready: Includes data caching and error-handling for seamless user experience.

🛠️ Tech Stack
•	Frontend: Streamlit (Web Framework)
•	Visualizations: Plotly & Seaborn
•	Data Processing: Pandas & NumPy
•	Machine Learning: Scikit-Learn
•	RandomForestClassifier
•	RandomForestRegressor
•	LabelEncoder

📂 Dataset Structure
The system is optimized for a comprehensive placement dataset. Key features include:
Feature	Description
ssc_percentage	10th Grade Score
hsc_percentage	12th Grade Score
degree_percentage	Undergraduate Score
work_experience	Months of previous experience
technical_skills_score	AI/Coding assessment score
specialization	Domain (Data Science, HR, Finance, etc.)
placed	Target (Class): 0 = No, 1 = Yes
salary_lpa	Target (Reg): Annual CTC in Lakhs

⚙️ Installation & Setup
1.	Clone the Repository:
Bash
git clone https://github.com/your-username/placement-prediction-system.git
cd placement-prediction-system
2.	Install Dependencies:
Bash
pip install -r requirements.txt
(Note: If you don't have a requirements file, install: pip install streamlit pandas numpy scikit-learn plotly)
3.	Prepare Data: Ensure your campus_placement_data.csv is in the root directory.
4.	Launch the App:
Bash
streamlit run app.py

🧠 Model Logic
The system follows a tiered prediction logic:
1.	Phase 1: The user inputs their profile.
2.	Phase 2: The Classification model calculates a Placement Probability Gauge.
3.	Phase 3: If the probability is , the Regression model activates to forecast the Salary Bracket.
4.	Phase 4: The model displays Feature Importance, showing the user which input had the most significant effect on their result.

🤝 Contributing
Contributions are welcome! If you'd like to improve the model accuracy or add new visualization tabs:
1.	Fork the Project.
2.	Create your Feature Branch (git checkout -b feature/AmazingFeature).
3.	Commit your Changes (git commit -m 'Add some AmazingFeature').
4.	Push to the Branch (git push origin feature/AmazingFeature).
5.	Open a Pull Request.
Developed with ❤️ for Students and Career Counselors.
