"""
Módulo para extraer características meta de datasets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List
from scipy import stats
from scipy.stats import entropy


def extract_statistical_features(X: np.ndarray, y: np.ndarray = None) -> Dict:
    """
    Extrae características estadísticas básicas del dataset.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas (opcional)
    
    Returns:
        Diccionario con características estadísticas
    """
    features = {}
    
    # Características básicas
    features['n_samples'] = X.shape[0]
    features['n_features'] = X.shape[1]
    features['n_numerical_features'] = X.select_dtypes(include=[np.number]).shape[1] if isinstance(X, pd.DataFrame) else X.shape[1]
    
    # Estadísticas de características numéricas
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])
    else:
        X_num = X
    
    if X_num.shape[1] > 0:
        features['mean_mean'] = np.mean(X_num.mean(axis=0))
        features['mean_std'] = np.mean(X_num.std(axis=0))
        features['mean_skewness'] = np.mean([stats.skew(X_num[:, i]) for i in range(X_num.shape[1])])
        features['mean_kurtosis'] = np.mean([stats.kurtosis(X_num[:, i]) for i in range(X_num.shape[1])])
    
    # Características de la variable objetivo
    if y is not None:
        features['n_classes'] = len(np.unique(y))
        class_counts = np.bincount(y.astype(int))
        features['class_entropy'] = entropy(class_counts / class_counts.sum())
        features['class_imbalance_ratio'] = np.min(class_counts) / np.max(class_counts) if len(class_counts) > 1 else 1.0
    
    return features


def extract_complexity_features(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Extrae características de complejidad del dataset.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas
    
    Returns:
        Diccionario con características de complejidad
    """
    features = {}
    
    # Ratio de dimensionalidad
    features['dimensionality_ratio'] = X.shape[1] / X.shape[0] if X.shape[0] > 0 else 0
    
    # Correlación promedio entre características
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])
    else:
        X_num = X
    
    if X_num.shape[1] > 1:
        corr_matrix = np.corrcoef(X_num.T)
        # Excluir diagonal
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        features['mean_correlation'] = np.mean(np.abs(corr_matrix[mask]))
        features['max_correlation'] = np.max(np.abs(corr_matrix[mask]))
    
    # Separabilidad (distancia promedio entre clases)
    if y is not None and len(np.unique(y)) > 1:
        unique_classes = np.unique(y)
        class_centroids = []
        for cls in unique_classes:
            class_mask = y == cls
            class_centroids.append(np.mean(X_num[class_mask], axis=0))
        
        if len(class_centroids) > 1:
            distances = []
            for i in range(len(class_centroids)):
                for j in range(i+1, len(class_centroids)):
                    distances.append(np.linalg.norm(class_centroids[i] - class_centroids[j]))
            features['inter_class_distance'] = np.mean(distances)
    
    return features


def extract_meta_features(dataset: Dict) -> Dict:
    """
    Extrae todas las características meta de un dataset.
    
    Args:
        dataset: Diccionario con dataset (de load_openml_dataset)
    
    Returns:
        Diccionario con todas las características meta
    """
    X = dataset['X']
    y = dataset['y']
    
    meta_features = {}
    
    # Añadir ID y nombre
    meta_features['dataset_id'] = dataset['id']
    meta_features['dataset_name'] = dataset['name']
    
    # Extraer características estadísticas
    stat_features = extract_statistical_features(X, y)
    meta_features.update(stat_features)
    
    # Extraer características de complejidad
    if y is not None:
        complexity_features = extract_complexity_features(X, y)
        meta_features.update(complexity_features)
    
    # Añadir metadatos de OpenML si están disponibles
    if 'metadata' in dataset:
        meta_features.update(dataset['metadata'])
    
    return meta_features


def extract_meta_features_batch(datasets: List[Dict]) -> pd.DataFrame:
    """
    Extrae características meta de múltiples datasets.
    
    Args:
        datasets: Lista de diccionarios con datasets
    
    Returns:
        DataFrame con características meta de todos los datasets
    """
    all_meta_features = []
    
    for dataset in datasets:
        try:
            meta_features = extract_meta_features(dataset)
            all_meta_features.append(meta_features)
        except Exception as e:
            print(f"Error extrayendo características meta del dataset {dataset.get('id', 'unknown')}: {e}")
    
    return pd.DataFrame(all_meta_features)

