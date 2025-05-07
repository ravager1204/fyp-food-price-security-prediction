# 📊 Forecasting Food Prices & Food Security in Malaysia

This Final Year Project builds predictive models to estimate monthly food prices and a Food Security Index (FSI) based on climate and agricultural data. It includes an interactive Streamlit dashboard to assist policymakers, researchers, and stakeholders in understanding trends and making informed decisions.

---

## 👨‍🎓 Author

**Wan Ahmad Fahim Munir Bin Wan Zaki**  
Bachelor of Computer Science (Data Analytics)  
Asia Pacific University of Technology & Innovation (APU)

---

## 🎯 Objectives

- Predict monthly **median prices** of key food items (in RM)
- Forecast **Food Security Index (FSI)** using climate and production data
- Visualize trends with an interactive **Streamlit UI**
- Allow manual and batch predictions via CSV

---

## 🧠 Models Used

- ✅ Random Forest *(Baseline)*
- ✅ XGBoost
- ✅ LightGBM
- ✅ LSTM
- ✅ Support Vector Regression (SVR)

### 📊 Evaluation Metrics

| Target            | Metrics            |
|------------------|---------------------|
| Food Prices (RM) | MAPE, RMSE          |
| FSI              | MAE, RMSE           |

---

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fyp-food-price-prediction.git
cd fyp-food-price-prediction


2. **Set up virtual environment**
python -m venv .venv
# Activate:
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows


3. Install dependencies
pip install -r requirements.txt


4. Run the Streamlit app
cd streamlit_app
streamlit run main.py

---

🧾 Data Sources
Department of Statistics Malaysia

World Bank Climate API

FAO Food Security Reports




---

## 🧾  Git Command Cheatsheet (for Daily Use)

| Task                         | Command |
|-----------------------------|---------|
| Check Git version           | `git --version` |
| Initialize repository       | `git init` |
| Check file status           | `git status` |
| Add files to staging        | `git add .` |
| Commit changes              | `git commit -m "message"` |
| View commit history         | `git log` |
| Connect to GitHub repo      | `git remote add origin <url>` |
| Push to remote              | `git push -u origin main` |
| Pull latest changes         | `git pull` |
| View branches               | `git branch` |
| Switch branch               | `git checkout branch-name` |
| Create new branch           | `git checkout -b new-branch` |
| Merge a branch              | `git merge branch-name` |
| Discard file changes        | `git restore filename.py` |
| Undo last commit (keep files) | `git reset --soft HEAD~1` |

---

## 🌿 3. Create and Push a `dev` Branch

```bash
git checkout -b dev
git push -u origin dev
