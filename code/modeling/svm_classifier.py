import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def train_svm_classifier(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple[SVC, dict]:
    """
    训练SVM分类器并返回模型和评估结果
    
    Args:
        X: 特征数据
        y: 目标变量
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        tuple:
            model: 训练好的SVM模型
            results: 包含评估指标的字典
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    
    # 网格搜索
    svm = SVC(probability=True)  # 添加probability=True以支持概率预测
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在测试集上评估
    y_pred = best_model.predict(X_test_scaled)
    
    # 计算评估指标
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred, normalize=["all"], labels=["Cu", "Au"]),
        'test_accuracy': best_model.score(X_test_scaled, y_test)
    }
    
    return best_model, results