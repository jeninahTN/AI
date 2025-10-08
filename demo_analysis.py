"""
Demo Script for Student Performance Analysis
This script demonstrates the complete analysis using sample data
"""

from student_performance_analysis import StudentPerformanceAnalysis
import os

def main():
    print("🎓 STUDENT PERFORMANCE ANALYSIS DEMO")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = StudentPerformanceAnalysis()
    
    # Check if UCI dataset exists
    dataset_file = "student-mat.csv"
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file '{dataset_file}' not found!")
        print("Please ensure the UCI Student Performance dataset is in the current directory.")
        print("Download from: https://archive.ics.uci.edu/dataset/320/student+performance")
        return
    
    print(f"📊 Using UCI Student Performance dataset: {dataset_file}")
    print("This is the complete analysis using the real UCI Student Performance data.")
    print()
    
    # Run the complete analysis
    success = analyzer.run_complete_analysis(dataset_file)
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("\nFor additional analysis options:")
        print("1. Check the generated PNG visualizations in the current directory")
        print("2. Review the comprehensive results and recommendations above")
        print("3. Use 'student-por.csv' for Portuguese language course analysis")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()