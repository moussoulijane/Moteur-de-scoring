"""
Calcul complet des métriques de performance
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calcule toutes les métriques de performance"""

    def __init__(self):
        self.metrics = {}
        self.confusion_mat = None

    def calculate_all_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calcule toutes les métriques

        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions binaires
            y_prob: Probabilités prédites (optionnel)
        """
        # Métriques de base
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        self.metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Matrice de confusion
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = self.confusion_mat.ravel()

        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        self.metrics['true_positives'] = int(tp)

        # Spécificité
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Métriques basées sur probabilités
        if y_prob is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            self.metrics['pr_auc'] = average_precision_score(y_true, y_prob)

        return self.metrics

    def print_metrics(self, title="MÉTRIQUES DE PERFORMANCE"):
        """Affiche les métriques"""
        print(f"\n📊 {title}")
        print("=" * 60)

        if 'accuracy' in self.metrics:
            print(f"  Accuracy:    {self.metrics['accuracy']:.4f}")
        if 'precision' in self.metrics:
            print(f"  Precision:   {self.metrics['precision']:.4f}")
        if 'recall' in self.metrics:
            print(f"  Recall:      {self.metrics['recall']:.4f}")
        if 'f1_score' in self.metrics:
            print(f"  F1-Score:    {self.metrics['f1_score']:.4f}")
        if 'specificity' in self.metrics:
            print(f"  Specificity: {self.metrics['specificity']:.4f}")

        if 'roc_auc' in self.metrics:
            print(f"\n  ROC-AUC:     {self.metrics['roc_auc']:.4f}")
        if 'pr_auc' in self.metrics:
            print(f"  PR-AUC:      {self.metrics['pr_auc']:.4f}")

        if self.confusion_mat is not None:
            print(f"\n  Matrice de confusion:")
            print(f"    TN: {self.metrics['true_negatives']:6d}  |  FP: {self.metrics['false_positives']:6d}")
            print(f"    FN: {self.metrics['false_negatives']:6d}  |  TP: {self.metrics['true_positives']:6d}")

    def plot_confusion_matrix(self, save_path=None):
        """Trace la matrice de confusion"""
        if self.confusion_mat is None:
            print("⚠️  Pas de matrice de confusion disponible")
            return

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Non Fondée', 'Fondée'],
            yticklabels=['Non Fondée', 'Fondée'],
            cbar=True
        )
        plt.xlabel('Prédiction', fontsize=12)
        plt.ylabel('Vérité', fontsize=12)
        plt.title('Matrice de Confusion', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """Trace la courbe ROC"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aléatoire')

        plt.xlabel('Taux de Faux Positifs', fontsize=12)
        plt.ylabel('Taux de Vrais Positifs', fontsize=12)
        plt.title('Courbe ROC', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def plot_precision_recall_curve(self, y_true, y_prob, save_path=None):
        """Trace la courbe Precision-Recall"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Courbe Precision-Recall', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def compare_metrics(self, metrics_2024, metrics_2025):
        """
        Compare les métriques entre 2024 et 2025

        Args:
            metrics_2024: Dict des métriques 2024
            metrics_2025: Dict des métriques 2025
        """
        print(f"\n📊 COMPARAISON 2024 vs 2025")
        print("=" * 70)
        print(f"{'Métrique':<20s} {'2024':>12s} {'2025':>12s} {'Δ':>12s} {'Δ%':>12s}")
        print("-" * 70)

        keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']

        for key in keys:
            if key in metrics_2024 and key in metrics_2025:
                val_2024 = metrics_2024[key]
                val_2025 = metrics_2025[key]
                delta = val_2025 - val_2024
                delta_pct = (delta / val_2024) * 100 if val_2024 != 0 else 0

                delta_str = f"{delta:+.4f}"
                delta_pct_str = f"{delta_pct:+.2f}%"

                # Coloriser (texte seulement)
                status = "✅" if abs(delta_pct) < 5 else "⚠️"

                print(f"{key:<20s} {val_2024:>12.4f} {val_2025:>12.4f} {delta_str:>12s} {delta_pct_str:>12s} {status}")

        print("-" * 70)

        # Analyse
        accuracy_degradation = ((metrics_2025['accuracy'] - metrics_2024['accuracy']) / metrics_2024['accuracy']) * 100
        if abs(accuracy_degradation) < 2:
            print("✅ STABILITÉ EXCELLENTE: Dégradation < 2%")
        elif abs(accuracy_degradation) < 5:
            print("✅ STABILITÉ ACCEPTABLE: Dégradation < 5%")
        else:
            print(f"❌ ALERTE: Dégradation > 5% ({accuracy_degradation:+.2f}%)")
            print("   Recommandation: Réentraînement du modèle sur données récentes")

    def get_metrics_dataframe(self):
        """Retourne les métriques sous forme de DataFrame"""
        return pd.DataFrame([self.metrics])
