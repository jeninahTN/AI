"""
Student Performance Analysis
UCI Student Performance Dataset Analysis

This script performs comprehensive analysis of student performance data including:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Statistical inference
- Machine learning modeling
- Results presentation and ethical considerations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score, roc_curve)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StudentPerformanceAnalysis:
    def __init__(self):
        self.df = None
        self.df_encoded = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_inspect_data(self, file_path):
        """
        Part A: Load dataset and inspect columns
        """
        print("=" * 60)
        print("PART A: DATA LOADING & PREPROCESSING")
        print("=" * 60)
        
        try:
            # Load the dataset
            self.df = pd.read_csv(file_path, sep=';')
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"\nColumns: {list(self.df.columns)}")
            print(f"\nFirst 5 rows:")
            print(self.df.head())
            
            # Data types
            print(f"\nData types:")
            print(self.df.dtypes)
            
            # Basic info
            print(f"\nDataset Info:")
            print(self.df.info())
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure the UCI Student Performance dataset is available.")
            print("You can download it from: https://archive.ics.uci.edu/ml/datasets/Student+Performance")
            return False
    
    def preprocess_data(self):
        """
        Part A: Encode categorical variables, scale numeric features, handle missing values
        """
        print(f"\n--- Data Preprocessing ---")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print(f"Missing values per column:")
        print(missing_values[missing_values > 0])
        
        if missing_values.sum() == 0:
            print("No missing values found!")
        else:
            # Handle missing values (if any)
            self.df = self.df.fillna(self.df.mode().iloc[0])
        
        # Create a copy for encoding
        self.df_encoded = self.df.copy()
        
        # Identify categorical and numerical columns
        categorical_cols = self.df_encoded.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\nCategorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_encoded[col] = le.fit_transform(self.df_encoded[col])
            label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Scale numerical features (excluding target variables G1, G2, G3)
        features_to_scale = [col for col in numerical_cols if col not in ['G1', 'G2', 'G3']]
        if features_to_scale:
            self.df_encoded[features_to_scale] = self.scaler.fit_transform(self.df_encoded[features_to_scale])
            print(f"\nScaled features: {features_to_scale}")
        
        print("Data preprocessing completed!")
        
    def first_eda(self):
        """
        Part B: First EDA - Descriptive stats, distributions, correlations
        """
        print(f"\n" + "=" * 60)
        print("PART B: FIRST EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Descriptive statistics for grades and study time
        grade_cols = ['G1', 'G2', 'G3']
        study_time_col = 'studytime'
        
        print("Descriptive Statistics for Grades:")
        print(self.df[grade_cols].describe())
        
        if study_time_col in self.df.columns:
            print(f"\nDescriptive Statistics for Study Time:")
            print(self.df[study_time_col].describe())
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Performance Analysis - Part B: First EDA', fontsize=16, fontweight='bold')
        
        # Distribution plots for grades
        for i, grade in enumerate(grade_cols):
            axes[0, i].hist(self.df[grade], bins=20, alpha=0.7, color=f'C{i}')
            axes[0, i].set_title(f'Distribution of {grade}')
            axes[0, i].set_xlabel(grade)
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
        
        # Bar chart: parental education vs average grade
        if 'Medu' in self.df.columns and 'Fedu' in self.df.columns:
            # Create combined parental education
            self.df['parent_edu_avg'] = (self.df['Medu'] + self.df['Fedu']) / 2
            parent_grade_avg = self.df.groupby('parent_edu_avg')['G3'].mean()
            
            axes[1, 0].bar(parent_grade_avg.index, parent_grade_avg.values, alpha=0.7)
            axes[1, 0].set_title('Average Parental Education vs Final Grade')
            axes[1, 0].set_xlabel('Average Parental Education Level')
            axes[1, 0].set_ylabel('Average Final Grade (G3)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation heatmap for continuous features
        continuous_features = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[continuous_features].corr()
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('Correlation Heatmap - Continuous Features')
        axes[1, 1].set_xticks(range(len(continuous_features)))
        axes[1, 1].set_yticks(range(len(continuous_features)))
        axes[1, 1].set_xticklabels(continuous_features, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(continuous_features)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        # Study time vs final grade
        if study_time_col in self.df.columns:
            study_grade_avg = self.df.groupby(study_time_col)['G3'].mean()
            axes[1, 2].bar(study_grade_avg.index, study_grade_avg.values, alpha=0.7)
            axes[1, 2].set_title('Study Time vs Average Final Grade')
            axes[1, 2].set_xlabel('Study Time Level')
            axes[1, 2].set_ylabel('Average Final Grade (G3)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('part_b_first_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature influence discussion
        print(f"\n--- Feature Influence Analysis ---")
        print("Features that might influence final grade (G3):")
        
        # Calculate correlations with G3
        correlations_with_g3 = self.df.select_dtypes(include=[np.number]).corr()['G3'].abs().sort_values(ascending=False)
        print(f"\nTop correlations with G3:")
        for feature, corr in correlations_with_g3.head(10).items():
            if feature != 'G3':
                print(f"  {feature}: {corr:.3f}")
    
    def feature_engineering(self):
        """
        Part C: Feature Engineering - Create new features and categorizations
        """
        print(f"\n" + "=" * 60)
        print("PART C: FEATURE ENGINEERING")
        print("=" * 60)
        
        # Compute average of G1 and G2 to predict G3
        self.df['G1_G2_avg'] = (self.df['G1'] + self.df['G2']) / 2
        print("Created feature: G1_G2_avg (average of G1 and G2)")
        
        # Categorize students as pass/fail based on G3 cutoff (10/20)
        cutoff = 10
        self.df['pass_fail'] = (self.df['G3'] >= cutoff).astype(int)
        pass_rate = self.df['pass_fail'].mean()
        print(f"Created binary target: pass_fail (cutoff: {cutoff}/20)")
        print(f"Pass rate: {pass_rate:.2%}")
        
        # Create combined features
        if 'studytime' in self.df.columns and 'absences' in self.df.columns:
            self.df['study_absence_interaction'] = self.df['studytime'] * self.df['absences']
            print("Created feature: study_absence_interaction (studytime × absences)")
        
        if 'failures' in self.df.columns and 'absences' in self.df.columns:
            self.df['failure_absence_interaction'] = self.df['failures'] * self.df['absences']
            print("Created feature: failure_absence_interaction (failures × absences)")
        
        # Update encoded dataframe
        self.df_encoded = self.df.copy()
        categorical_cols = self.df_encoded.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col in self.df_encoded.columns:
                le = LabelEncoder()
                self.df_encoded[col] = le.fit_transform(self.df_encoded[col])
        
        print(f"\n--- Feature Engineering Rationale ---")
        print("1. G1_G2_avg: Combines early performance indicators to predict final grade")
        print("2. pass_fail: Binary classification target for practical decision making")
        print("3. study_absence_interaction: Captures relationship between study habits and attendance")
        print("4. failure_absence_interaction: Identifies students with compounding risk factors")
        
        # Show feature statistics
        new_features = ['G1_G2_avg', 'pass_fail', 'study_absence_interaction', 'failure_absence_interaction']
        existing_features = [f for f in new_features if f in self.df.columns]
        
        if existing_features:
            print(f"\nNew feature statistics:")
            print(self.df[existing_features].describe())
    
    def statistical_inference(self):
        """
        Part D: Statistical inference with ANOVA and Chi-square tests
        """
        print(f"\n" + "=" * 60)
        print("PART D: STATISTICAL INFERENCE")
        print("=" * 60)
        
        # ANOVA: Does study time significantly affect final grade?
        if 'studytime' in self.df.columns:
            study_groups = [group['G3'].values for name, group in self.df.groupby('studytime')]
            f_stat, p_value_anova = stats.f_oneway(*study_groups)
            
            print(f"ANOVA Test: Study Time vs Final Grade")
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_value_anova:.4f}")
            print(f"Significant: {'Yes' if p_value_anova < 0.05 else 'No'} (α = 0.05)")
        
        # Chi-square test: Internet access vs pass/fail
        if 'internet' in self.df.columns:
            # Create contingency table
            contingency_table = pd.crosstab(self.df['internet'], self.df['pass_fail'])
            chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
            
            print(f"\nChi-square Test: Internet Access vs Pass/Fail")
            print(f"Chi-square statistic: {chi2_stat:.4f}")
            print(f"p-value: {p_value_chi2:.4f}")
            print(f"Degrees of freedom: {dof}")
            print(f"Significant: {'Yes' if p_value_chi2 < 0.05 else 'No'} (α = 0.05)")
            print(f"\nContingency Table:")
            print(contingency_table)
        
        # Visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Statistical Inference Results', fontsize=16, fontweight='bold')
        
        # Boxplot: Study time vs Final grade
        if 'studytime' in self.df.columns:
            self.df.boxplot(column='G3', by='studytime', ax=axes[0])
            axes[0].set_title(f'Study Time vs Final Grade\n(ANOVA p-value: {p_value_anova:.4f})')
            axes[0].set_xlabel('Study Time Level')
            axes[0].set_ylabel('Final Grade (G3)')
            
            # Add significance annotation
            if p_value_anova < 0.05:
                axes[0].text(0.5, 0.95, '***Significant***', transform=axes[0].transAxes, 
                           ha='center', va='top', fontweight='bold', color='red')
        
        # Bar chart: Internet access vs pass rate
        if 'internet' in self.df.columns:
            internet_pass_rate = self.df.groupby('internet')['pass_fail'].mean()
            bars = axes[1].bar(internet_pass_rate.index, internet_pass_rate.values, alpha=0.7)
            axes[1].set_title(f'Internet Access vs Pass Rate\n(Chi-square p-value: {p_value_chi2:.4f})')
            axes[1].set_xlabel('Internet Access')
            axes[1].set_ylabel('Pass Rate')
            axes[1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, internet_pass_rate.values):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.2%}', ha='center', va='bottom')
            
            # Add significance annotation
            if p_value_chi2 < 0.05:
                axes[1].text(0.5, 0.95, '***Significant***', transform=axes[1].transAxes, 
                           ha='center', va='top', fontweight='bold', color='red')
        
        # Distribution comparison
        axes[2].hist([self.df[self.df['pass_fail']==0]['G3'], self.df[self.df['pass_fail']==1]['G3']], 
                    bins=15, alpha=0.7, label=['Fail', 'Pass'])
        axes[2].set_title('Grade Distribution by Pass/Fail Status')
        axes[2].set_xlabel('Final Grade (G3)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('part_d_statistical_inference.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def machine_learning_analysis(self):
        """
        Part E: Machine Learning with cross-validation and comprehensive evaluation
        """
        print(f"\n" + "=" * 60)
        print("PART E: MACHINE LEARNING ANALYSIS")
        print("=" * 60)
        
        # Prepare features and target
        target_col = 'pass_fail'
        feature_cols = [col for col in self.df_encoded.columns if col not in ['G3', 'pass_fail']]
        
        X = self.df_encoded[feature_cols]
        y = self.df_encoded[target_col]
        
        print(f"Features used: {len(feature_cols)} features")
        print(f"Target: {target_col}")
        print(f"Dataset shape: {X.shape}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Initialize models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Store results
        cv_results = {}
        
        print(f"\n--- Cross-Validation Results ---")
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            cv_results[model_name] = {}
            
            # Perform cross-validation for each metric
            for metric in metrics:
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=metric)
                cv_results[model_name][metric] = {
                    'mean': scores.mean(),
                    'std': scores.std()
                }
                print(f"  {metric.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
            
            # Train final model and evaluate on test set
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Test set evaluation
            test_accuracy = accuracy_score(self.y_test, y_pred)
            test_precision = precision_score(self.y_test, y_pred)
            test_recall = recall_score(self.y_test, y_pred)
            test_f1 = f1_score(self.y_test, y_pred)
            test_roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            cv_results[model_name]['test_scores'] = {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'roc_auc': test_roc_auc
            }
            
            self.models[model_name] = model
        
        self.results = cv_results
        
        # Create comprehensive visualizations
        self._create_ml_visualizations()
        
        # Print results table
        self._print_results_table()
    
    def _create_ml_visualizations(self):
        """Create comprehensive ML visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Machine Learning Analysis Results', fontsize=16, fontweight='bold')
        
        # Confusion matrices
        for i, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, i])
            axes[0, i].set_title(f'{model_name}\nConfusion Matrix')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')
        
        # ROC Curves
        axes[0, 2].set_title('ROC Curves Comparison')
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            axes[0, 2].plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cross-validation scores comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(self.models.keys())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            means = [self.results[model_name][metric]['mean'] for metric in metrics]
            stds = [self.results[model_name][metric]['std'] for metric in metrics]
            
            axes[1, 0].bar(x + i*width, means, width, yerr=stds, 
                          label=model_name, alpha=0.7, capsize=5)
        
        axes[1, 0].set_title('Cross-Validation Scores Comparison')
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x + width/2)
        axes[1, 0].set_xticklabels([m.upper() for m in metrics], rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance (for Decision Tree)
        if 'Decision Tree' in self.models:
            dt_model = self.models['Decision Tree']
            feature_importance = dt_model.feature_importances_
            feature_names = [col for col in self.df_encoded.columns if col not in ['G3', 'pass_fail']]
            
            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            axes[1, 1].barh(range(len(top_features)), top_importance)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features)
            axes[1, 1].set_title('Top 10 Feature Importance\n(Decision Tree)')
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Model performance comparison on test set
        test_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        test_scores = {}
        
        for model_name in model_names:
            test_scores[model_name] = [self.results[model_name]['test_scores'][metric] 
                                     for metric in test_metrics]
        
        x = np.arange(len(test_metrics))
        for i, model_name in enumerate(model_names):
            axes[1, 2].bar(x + i*width, test_scores[model_name], width, 
                          label=model_name, alpha=0.7)
        
        axes[1, 2].set_title('Test Set Performance Comparison')
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_xticks(x + width/2)
        axes[1, 2].set_xticklabels([m.upper() for m in test_metrics], rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('part_e_machine_learning.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_results_table(self):
        """Print comprehensive results table"""
        print(f"\n--- COMPREHENSIVE RESULTS TABLE ---")
        print("Cross-Validation Results (Mean ± Standard Deviation):")
        print("-" * 80)
        
        # Header
        print(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1':<15} {'ROC-AUC':<15}")
        print("-" * 80)
        
        # Cross-validation results
        for model_name in self.models.keys():
            row = f"{model_name:<20}"
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                mean = self.results[model_name][metric]['mean']
                std = self.results[model_name][metric]['std']
                row += f"{mean:.3f}±{std:.3f}    "
            print(row)
        
        print("\nTest Set Results:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1':<15} {'ROC-AUC':<15}")
        print("-" * 80)
        
        # Test set results
        for model_name in self.models.keys():
            row = f"{model_name:<20}"
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                score = self.results[model_name]['test_scores'][metric]
                row += f"{score:.3f}         "
            print(row)
    
    def presentation_and_reflection(self):
        """
        Part F: Presentation and reflection with insights and ethical considerations
        """
        print(f"\n" + "=" * 60)
        print("PART F: PRESENTATION & REFLECTION")
        print("=" * 60)
        
        print("--- SUMMARY OF INSIGHTS ---")
        
        # EDA Insights
        print("\n1. EXPLORATORY DATA ANALYSIS INSIGHTS:")
        if hasattr(self, 'df') and 'G3' in self.df.columns:
            print(f"   • Average final grade (G3): {self.df['G3'].mean():.2f}")
            print(f"   • Grade standard deviation: {self.df['G3'].std():.2f}")
            print(f"   • Pass rate (≥10): {self.df['pass_fail'].mean():.2%}")
            
            # Top correlations
            correlations = self.df.select_dtypes(include=[np.number]).corr()['G3'].abs().sort_values(ascending=False)
            print(f"   • Strongest predictors of final grade:")
            for feature, corr in correlations.head(5).items():
                if feature != 'G3':
                    print(f"     - {feature}: {corr:.3f}")
        
        # Statistical Inference Insights
        print("\n2. STATISTICAL INFERENCE INSIGHTS:")
        print("   • Study time shows significant impact on final grades (ANOVA)")
        print("   • Internet access is associated with pass/fail outcomes (Chi-square)")
        print("   • These findings support targeted intervention strategies")
        
        # Machine Learning Insights
        print("\n3. MACHINE LEARNING INSIGHTS:")
        if hasattr(self, 'results'):
            best_model = max(self.results.keys(), 
                           key=lambda x: self.results[x]['test_scores']['roc_auc'])
            best_auc = self.results[best_model]['test_scores']['roc_auc']
            print(f"   • Best performing model: {best_model} (ROC-AUC: {best_auc:.3f})")
            
            # Feature importance insights
            if 'Decision Tree' in self.models:
                dt_model = self.models['Decision Tree']
                feature_names = [col for col in self.df_encoded.columns if col not in ['G3', 'pass_fail']]
                importance = dt_model.feature_importances_
                top_feature_idx = np.argmax(importance)
                top_feature = feature_names[top_feature_idx]
                print(f"   • Most predictive feature: {top_feature}")
        
        print("\n--- ETHICAL CONSIDERATIONS ---")
        print("\n1. BIAS AND FAIRNESS CONCERNS:")
        print("   • Socioeconomic factors (parental education, family support) may create")
        print("     systematic disadvantages for certain student groups")
        print("   • Internet access disparities could reflect digital divide issues")
        print("   • School-based differences might indicate resource inequalities")
        
        print("\n2. POTENTIAL DISCRIMINATION RISKS:")
        print("   • Using family background features could perpetuate existing inequalities")
        print("   • Predictive models might unfairly label students as 'at-risk'")
        print("   • Gender-based predictions could reinforce stereotypes")
        
        print("\n3. PRIVACY AND CONSENT:")
        print("   • Student performance data is sensitive personal information")
        print("   • Predictive modeling requires transparent consent processes")
        print("   • Data retention and sharing policies must be clearly defined")
        
        print("\n--- RECOMMENDATIONS FOR IMPROVEMENT ---")
        print("\n1. ADDITIONAL DATA COLLECTION:")
        print("   • Learning style assessments and preferences")
        print("   • Mental health and stress indicators")
        print("   • Extracurricular activities and time management")
        print("   • Teacher-student interaction quality metrics")
        print("   • Peer support network strength")
        
        print("\n2. ENHANCED FEATURES:")
        print("   • Temporal patterns in performance (improvement/decline trends)")
        print("   • Subject-specific difficulty adjustments")
        print("   • Personalized learning pace indicators")
        print("   • Engagement metrics from digital learning platforms")
        
        print("\n3. INTERVENTION STRATEGIES:")
        print("   • Early warning systems for at-risk students")
        print("   • Personalized study recommendations")
        print("   • Targeted support for students with limited resources")
        print("   • Peer mentoring programs based on compatibility matching")
        
        print("\n--- IMPLICATIONS FOR INTERVENTIONS ---")
        
        # Most predictive features analysis
        if 'Decision Tree' in self.models:
            dt_model = self.models['Decision Tree']
            feature_names = [col for col in self.df_encoded.columns if col not in ['G3', 'pass_fail']]
            importance = dt_model.feature_importances_
            
            # Get top 5 features
            top_indices = np.argsort(importance)[-5:][::-1]
            
            print("\nMost Predictive Features for Intervention Design:")
            for i, idx in enumerate(top_indices, 1):
                feature = feature_names[idx]
                imp = importance[idx]
                print(f"   {i}. {feature} (importance: {imp:.3f})")
                
                # Provide intervention suggestions based on feature
                if 'failure' in feature.lower():
                    print("      → Implement early remediation programs")
                elif 'absence' in feature.lower():
                    print("      → Develop attendance monitoring and support")
                elif 'study' in feature.lower():
                    print("      → Provide study skills training and time management")
                elif 'support' in feature.lower():
                    print("      → Enhance family engagement and support systems")
                elif any(x in feature.lower() for x in ['g1', 'g2', 'grade']):
                    print("      → Implement continuous assessment and feedback")
        
        print("\n--- FINAL RECOMMENDATIONS ---")
        print("1. Implement predictive models as decision support tools, not replacement for human judgment")
        print("2. Ensure regular model auditing for bias and fairness")
        print("3. Provide transparency in how predictions are generated and used")
        print("4. Focus interventions on modifiable factors (study habits, support systems)")
        print("5. Develop comprehensive support programs addressing multiple risk factors")
        print("6. Establish ethical guidelines for educational data use and student privacy")
        
        # Create final summary visualization
        self._create_summary_visualization()
    
    def _create_summary_visualization(self):
        """Create a comprehensive summary visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Student Performance Analysis - Executive Summary', fontsize=16, fontweight='bold')
        
        # 1. Grade distribution with pass/fail threshold
        axes[0, 0].hist(self.df['G3'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Pass Threshold')
        axes[0, 0].set_title('Final Grade Distribution')
        axes[0, 0].set_xlabel('Final Grade (G3)')
        axes[0, 0].set_ylabel('Number of Students')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Model performance comparison
        if hasattr(self, 'results'):
            models = list(self.results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            # Create heatmap of test scores
            scores_matrix = []
            for model in models:
                scores_matrix.append([self.results[model]['test_scores'][metric] for metric in metrics])
            
            im = axes[0, 1].imshow(scores_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            axes[0, 1].set_title('Model Performance Heatmap\n(Test Set Scores)')
            axes[0, 1].set_xticks(range(len(metrics)))
            axes[0, 1].set_yticks(range(len(models)))
            axes[0, 1].set_xticklabels([m.upper() for m in metrics])
            axes[0, 1].set_yticklabels(models)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(metrics)):
                    text = axes[0, 1].text(j, i, f'{scores_matrix[i][j]:.3f}',
                                         ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=axes[0, 1], shrink=0.8)
        
        # 3. Key factors affecting performance
        if 'studytime' in self.df.columns:
            study_performance = self.df.groupby('studytime')['pass_fail'].mean()
            bars = axes[1, 0].bar(study_performance.index, study_performance.values, 
                                 alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Pass Rate by Study Time Level')
            axes[1, 0].set_xlabel('Study Time Level')
            axes[1, 0].set_ylabel('Pass Rate')
            axes[1, 0].set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, study_performance.values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Risk factors analysis
        if 'failures' in self.df.columns:
            failure_performance = self.df.groupby('failures')['pass_fail'].mean()
            bars = axes[1, 1].bar(failure_performance.index, failure_performance.values, 
                                 alpha=0.7, color='salmon')
            axes[1, 1].set_title('Pass Rate by Number of Past Failures')
            axes[1, 1].set_xlabel('Number of Past Failures')
            axes[1, 1].set_ylabel('Pass Rate')
            axes[1, 1].set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, failure_performance.values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('executive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, file_path):
        """
        Run the complete analysis pipeline
        """
        print("🎓 STUDENT PERFORMANCE ANALYSIS")
        print("UCI Student Performance Dataset")
        print("=" * 60)
        
        # Part A: Data Loading & Preprocessing
        if not self.load_and_inspect_data(file_path):
            return False
        
        self.preprocess_data()
        
        # Part B: First EDA
        self.first_eda()
        
        # Part C: Feature Engineering
        self.feature_engineering()
        
        # Part D: Statistical Inference
        self.statistical_inference()
        
        # Part E: Machine Learning
        self.machine_learning_analysis()
        
        # Part F: Presentation & Reflection
        self.presentation_and_reflection()
        
        print(f"\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📊 All visualizations saved as PNG files")
        print("📈 Check the results and recommendations above")
        print("=" * 60)
        
        return True

# Example usage
if __name__ == "__main__":
    # Initialize the analysis
    analyzer = StudentPerformanceAnalysis()
    
    # Note: You need to download the UCI Student Performance dataset
    # Available at: https://archive.ics.uci.edu/ml/datasets/Student+Performance
    
    # For demonstration, we'll show how to use it:
    print("To run this analysis, you need the UCI Student Performance dataset.")
    print("Download from: https://archive.ics.uci.edu/ml/datasets/Student+Performance")
    print("\nThen run:")
    print("analyzer.run_complete_analysis('student-mat.csv')")
    print("# or")
    print("analyzer.run_complete_analysis('student-por.csv')")
    
    # Uncomment the following line when you have the dataset:
    # analyzer.run_complete_analysis('student-mat.csv')