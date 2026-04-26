import sys
import os
import subprocess

def check_env():
    print("--- PHASE 1: ENVIRONMENT SETUP & HEALTH CHECK ---")
    
    # 1. Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required.")
        return
    else:
        print(f"✅ Python {sys.version.split()[0]} detected.")

    # 2. Folder Structure
    folders = [
        "data/raw", "data/processed", "notebooks", "src", 
        "models", "outputs/figures", "outputs/reports", "outputs/dashboard"
    ]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"Created folder: {f}")
        else:
            print(f"Folder exists: {f}")

    # 3. Import Health Check
    libraries = [
        "pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy", "joblib"
    ]
    # Optional libraries depending on environment
    opt_libs = ["xgboost", "plotly", "streamlit"]
    
    print("\nVerifying Core Libraries:")
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✅ {lib} imported successfully.")
        except ImportError:
            print(f"❌ {lib} missing.")

    print("\nVerifying Optional Libraries:")
    for lib in opt_libs:
        try:
            __import__(lib)
            print(f"✅ {lib} available.")
        except ImportError:
            print(f"⚠️ {lib} not available (will use fallbacks).")

    print("\n--- PROBLEM STATEMENT SUMMARY ---")
    print("Goal: Transform raw e-commerce data (Olist) into actionable insights.")
    print("Tasks: RFM Segmentation, Sales Forecasting, and Strategic Dashboarding.")

if __name__ == "__main__":
    check_env()
