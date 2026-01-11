"""
Calibration des probabilités prédites
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ProbabilityCalibrator:
    """
    Calibre les probabilités d'un modèle
    Vérifie la qualité de calibration
    """

    def __init__(self, method='isotonic', cv_folds=5):
        """
        Args:
            method: 'isotonic' ou 'sigmoid'
            cv_folds: Nombre de folds pour calibration
        """
        self.method = method
        self.cv_folds = cv_folds
        self.calibrated_model = None
        self.calibration_metrics = {}

    def fit(self, model, X, y):
        """
        Calibre le modèle

        Args:
            model: Modèle déjà entraîné
            X: Features de validation
            y: Target de validation
        """
        print(f"\n🎯 CALIBRATION DES PROBABILITÉS")
        print("=" * 60)
        print(f"Méthode: {self.method}")
        print(f"CV Folds: {self.cv_folds}")

        # Calibration avec CV
        self.calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=self.method,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        )

        self.calibrated_model.fit(X, y)

        print("✅ Calibration terminée!")

        return self

    def evaluate_calibration(self, model, X, y, n_bins=10, plot=True, save_path=None):
        """
        Évalue la qualité de calibration

        Args:
            model: Modèle à évaluer (calibré ou non)
            X: Features
            y: Target
            n_bins: Nombre de bins pour la courbe de calibration
            plot: Afficher le graphique
            save_path: Chemin pour sauvegarder le graphique
        """
        # Prédictions
        y_prob = model.predict_proba(X)[:, 1]

        # Courbe de calibration
        fraction_positives, mean_predicted = calibration_curve(
            y, y_prob,
            n_bins=n_bins,
            strategy='uniform'
        )

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y, y_prob, n_bins=n_bins)

        # Brier score
        brier = np.mean((y_prob - y) ** 2)

        self.calibration_metrics = {
            'ece': ece,
            'brier_score': brier,
            'fraction_positives': fraction_positives,
            'mean_predicted': mean_predicted
        }

        print(f"\n📊 Métriques de calibration:")
        print(f"  - Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  - Brier Score: {brier:.4f}")

        if ece < 0.05:
            print(f"  ✅ Excellente calibration (ECE < 0.05)")
        elif ece < 0.10:
            print(f"  ✅ Bonne calibration (ECE < 0.10)")
        elif ece < 0.15:
            print(f"  ⚠️  Calibration moyenne (ECE < 0.15)")
        else:
            print(f"  ❌ Mauvaise calibration (ECE >= 0.15) - Recalibration recommandée!")

        # Graphique
        if plot:
            self._plot_calibration_curve(
                fraction_positives,
                mean_predicted,
                ece,
                brier,
                save_path
            )

        return self.calibration_metrics

    def _calculate_ece(self, y_true, y_prob, n_bins=10):
        """
        Calcule Expected Calibration Error

        ECE = Σ (|B_i| / n) * |acc(B_i) - conf(B_i)|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            # Indices dans le bin
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])

            if mask.sum() == 0:
                continue

            # Accuracy et confidence moyennes dans le bin
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_size = mask.sum()

            ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)

        return ece

    def _plot_calibration_curve(
        self,
        fraction_positives,
        mean_predicted,
        ece,
        brier,
        save_path=None
    ):
        """Trace la courbe de calibration"""
        plt.figure(figsize=(10, 8))

        # Courbe de calibration
        plt.plot(mean_predicted, fraction_positives, 's-', label='Modèle', linewidth=2)

        # Diagonale parfaite
        plt.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré', linewidth=2)

        plt.xlabel('Probabilité prédite moyenne', fontsize=12)
        plt.ylabel('Fraction de positifs', fontsize=12)
        plt.title(
            f'Courbe de Calibration\nECE: {ece:.4f} | Brier: {brier:.4f}',
            fontsize=14
        )
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  📊 Graphique sauvegardé: {save_path}")

        plt.close()

    def predict_proba(self, X):
        """Prédictions calibrées"""
        if self.calibrated_model is None:
            raise ValueError("Modèle non calibré. Appelez fit() d'abord.")

        return self.calibrated_model.predict_proba(X)

    def predict(self, X, threshold=0.5):
        """Prédictions binaires avec seuil"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
