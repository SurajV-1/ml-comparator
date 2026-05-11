# 🤖 ML Model Comparator

A full-stack interactive web app to **compare Linear, Polynomial, and SVR regression models** on various datasets — with live charts, metrics, and single-point predictions.

![Tech Stack](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue)
![Tech Stack](https://img.shields.io/badge/Backend-Flask%20%2B%20scikit--learn-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **4 Dataset Types** — Sine Wave, Quadratic, Cubic, Linear
- **3 ML Models** — Linear Regression, Polynomial Regression (tunable degree), SVR (tunable C, ε, kernel)
- **Interactive Hyperparameters** — Sliders and dropdowns for live tuning
- **Performance Metrics** — R², MSE, RMSE, MAE with color-coded progress bars
- **Point Prediction** — Enter any X value to get predictions from all 3 models instantly
- **Best Model Highlight** — Automatically shows which model wins by R² score
- **Dark Premium UI** — Glassmorphism cards, smooth animations, responsive design

---

## 🚀 Local Development

### Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 🌐 Deployment

### Backend → [Render.com](https://render.com) (Free)

| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |

### Frontend → [Vercel.com](https://vercel.com) (Free)

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |

> **After deploying backend**, update `API_URL` in `frontend/src/App.jsx` to your Render URL.

---

## 📁 Project Structure

```
ml-comparison/
├── backend/
│   ├── app.py            ← Flask REST API
│   ├── requirements.txt  ← Python deps
│   └── render.yaml       ← Render deploy config
└── frontend/
    ├── src/
    │   ├── App.jsx        ← Main React UI
    │   ├── index.css      ← Design system
    │   └── main.jsx       ← Entry point
    ├── index.html
    └── vercel.json        ← Vercel deploy config
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/train` | Train all 3 models, returns predictions + metrics |
| `POST` | `/api/predict` | Single-point prediction for a given X value |

---

## 📊 Models Compared

| Model | Strengths | Weaknesses |
|---|---|---|
| **Linear Regression** | Simple, fast, interpretable | Only fits linear relationships |
| **Polynomial Regression** | Captures non-linear patterns | Overfits at high degrees |
| **SVR** | Robust to outliers, flexible kernels | Slower, harder to tune |
