# 📧 Email Spam Detection Using Machine Learning

A machine learning pipeline that classifies emails as **spam** or **ham (legitimate)** using natural language processing and classical ML algorithms. Built and evaluated on a dataset of 83,000+ labeled emails.

## 🎯 Overview

This project implements a complete spam detection pipeline:
- Text preprocessing (cleaning, stopword removal, lemmatization)
- Feature extraction using TF-IDF vectorization
- Training and comparison of 5 machine learning classifiers
- Model persistence and real-time prediction on new emails

**Best model: Linear SVM — 98.61% accuracy, 0.9861 F1-score**

## 📊 Results

| Model | Accuracy | F1-Score | Training Time (s) |
|---|---|---|---|
| **Linear SVM** | **98.61%** | **0.9861** | 6.35 |
| Random Forest | 98.35% | 0.9835 | 374.06 |
| Logistic Regression | 98.33% | 0.9833 | 3.62 |
| Naive Bayes | 96.50% | 0.9650 | 2.85 |
| Decision Tree | 96.30% | 0.9630 | 162.74 |

## 🗂️ Project Structure