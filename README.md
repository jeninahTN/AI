# Student Performance Analysis

A comprehensive analysis of student performance using the UCI Student Performance Dataset. This project implements all six parts of the analysis as specified in the requirements.

## 📋 Project Overview

This project performs a complete analysis of student performance data including:

- **Part A**: Data Loading & Preprocessing (15 points)
- **Part B**: First EDA (20 points) 
- **Part C**: Feature Engineering (15 points)
- **Part D**: Second EDA & Statistical Inference (15 points)
- **Part E**: Simple Machine Learning (25 points)
- **Part F**: Presentation & Reflection (10 points)

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Demo

To see the analysis in action with sample data:

```bash
python demo_analysis.py
```

### Using Real UCI Data

1. **Download the UCI Student Performance Dataset**:
   - Visit: https://archive.ics.uci.edu/dataset/320/student+performance
   - Download either `student-mat.csv` (Mathematics) or `student-por.csv` (Portuguese)

2. **Run the analysis**:
   ```python
   from student_performance_analysis import StudentPerformanceAnalysis
   
   analyzer = StudentPerformanceAnalysis()
   analyzer.run_complete_analysis('student-mat.csv')
   ```

## 📊 Analysis Components

### Part A: Data Loading & Preprocessing
- ✅ Load dataset and inspect columns
- ✅ Encode categorical variables (school, sex, parental education, etc.)
- ✅ Scale numeric features (grades, study time, absences)
- ✅ Check for missing values and handle them

### Part B: First EDA
- ✅ Descriptive stats for grades and study time
- ✅ Distribution plots (histograms) for grades
- ✅ Bar charts: parental education vs average grade
- ✅ Correlation heatmap for continuous features
- ✅ Discussion of features that might influence final grade

### Part C: Feature Engineering
- ✅ Compute average of G1 and G2 to predict G3
- ✅ Categorize students as pass/fail based on G3 cutoff (10/20)
- ✅ Create combined features: study_time × absences, failures × absences
- ✅ Discuss rationale for each feature

### Part D: Second EDA & Statistical Inference
- ✅ ANOVA: Does study time significantly affect final grade?
- ✅ Chi-square test: Association between internet access and pass/fail
- ✅ Visualize results: boxplots, bar charts with significance annotations

### Part E: Simple Machine Learning
- ✅ Split data into train/test sets
- ✅ Train models: Decision Tree, Logistic Regression
- ✅ Evaluate: Confusion matrix, Accuracy, Precision, Recall, ROC curve
- ✅ Compute standard deviation of metrics across cross-validation folds
- ✅ Report mean ± std format for all metrics

### Part F: Presentation & Reflection
- ✅ Summarize insights from EDA and ML
- ✅ Discuss ethical considerations (bias, fairness)
- ✅ Suggest additional data/features for improvement
- ✅ Results table with mean ± std format
- ✅ Discuss most predictive features and intervention implications

## 📈 Output Files

The analysis generates several visualization files:

- `part_b_first_eda.png` - Initial exploratory data analysis
- `part_d_statistical_inference.png` - Statistical test results
- `part_e_machine_learning.png` - ML model performance
- `executive_summary.png` - Comprehensive summary visualization

## 🔍 Key Features

### Statistical Analysis
- **ANOVA testing** for study time impact on grades
- **Chi-square testing** for categorical associations
- **Correlation analysis** for feature relationships

### Machine Learning
- **Cross-validation** with 5-fold stratified sampling
- **Multiple metrics** with standard deviations
- **Feature importance** analysis
- **ROC curve** comparison

### Ethical Considerations
- Bias detection in socioeconomic factors
- Fairness assessment across student groups
- Privacy and consent considerations
- Intervention strategy recommendations

## 📋 Dataset Information

The UCI Student Performance Dataset contains information about student achievement in secondary education. Key attributes include:

- **Demographics**: age, sex, address, family size
- **Social**: parental education, family support, internet access
- **School**: study time, failures, absences, extra support
- **Grades**: G1 (first period), G2 (second period), G3 (final grade)

## 🎯 Results Summary

The analysis provides:

1. **Predictive Models**: Decision Tree and Logistic Regression with cross-validation
2. **Statistical Insights**: Significant factors affecting student performance
3. **Feature Importance**: Most influential variables for academic success
4. **Intervention Recommendations**: Data-driven suggestions for student support
5. **Ethical Guidelines**: Considerations for fair and responsible use

## 🔧 Technical Details

- **Python 3.7+** required
- **Key Libraries**: pandas, scikit-learn, matplotlib, seaborn, scipy
- **Cross-validation**: 5-fold stratified sampling
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Statistical Tests**: ANOVA, Chi-square

## 📝 License

This project is for educational purposes. The UCI Student Performance Dataset is publicly available for research use.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please refer to the analysis documentation within the code.

---

**Note**: This analysis demonstrates comprehensive data science methodology including preprocessing, EDA, feature engineering, statistical inference, machine learning, and ethical considerations.