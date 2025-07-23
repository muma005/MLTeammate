# 📈 MLTeammate – Project Progress Tracker

This document helps track the overall vision, completed milestones, current tasks, and upcoming goals for the MLTeammate project.  
Use this to maintain clarity, motivation, and alignment with the long-term roadmap.

---

## 🧩 Project Vision

MLTeammate is an **open-source AutoML teammate** designed to be:

- Transparent 🪞 (no black-boxes)
- Modular 🧱 (plug-in learners, searchers, metrics)
- Research-ready 🧪 (easy experiment tracking & customization)
- Beginner-friendly 🚀 (educational code structure & tutorials)

It aims to serve as a personal and collaborative ML companion.

---

## ✅ Completed So Far

### 🔧 Core Architecture
- [x] `AutoMLController` class
- [x] Learner abstraction (XGBoost, LightGBM)
- [x] Config space definition (`config_spaces.py`)
- [x] `OptunaSearch` class
- [x] Basic metrics module (`metrics.py`)
- [x] Training + `.score()` compatibility

### 📊 Evaluation + Utility
- [x] Cross-validation support
- [x] sklearn-like `.fit()` and `.score()` API
- [x] Internal logging with trial tracking
- [x] Functional test with synthetic data
- [x] `01_quickstart_basic.py` complete

### 📚 Documentation
- [x] Project `README.md`
- [x] This `PROJECT_PROGRESS.md` tracker
- [x] Clear project structure and examples

---

## 🔄 In Progress

### 🧪 Tutorials & Examples
- [ ] `02_with_cross_validation.py` (nearly done)
- [ ] `03_with_mlflow.py` (requires tracking + artifact support)
- [ ] `04_add_custom_learner.py` (custom learner API testing)
- [ ] `05_optuna_search_example.py` (demonstrating tuner flexibility)

### 📁 Docs
- [ ] `docs/installation.md`  
- [ ] `docs/getting_started.md`  
- [ ] `docs/extending.md` (for contributors)

---

## 📌 Next Steps

### ✍️ Core Features (High Priority)
- [ ] Add FLAMLSearch (optional fallback searcher)
- [ ] Add LightGBMLearner abstraction
- [ ] Allow user-defined scoring functions
- [ ] Handle multi-class and regression tasks better

### 🧪 Experiments
- [ ] Add MLflow experiment tracking support
- [ ] Enable saving/loading best models via `controller.save_model()`  

### 🧠 Smart Features (Phase 2)
- [ ] Add meta-learning support (transfer prior configs)
- [ ] Build leaderboard across runs
- [ ] Optional ensemble support

---

## 🗂️ Planned Project Structure (End Goal)

