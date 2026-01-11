"""
Générateur de données réalistes pour réclamations bancaires 2024 et 2025
Inclut un drift temporel pour simuler l'évolution des comportements
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class ReclamationDataGenerator:
    """Génère des données réalistes de réclamations avec drift temporel"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed

        # Métadonnées
        self.familles = {
            'Monétique': ['GAB', 'Carte bancaire', 'TPE'],
            'Crédit': ['Crédit personnel', 'Crédit immobilier', 'Crédit consommation'],
            'Frais bancaires': ['Tenue de compte', 'Commissions', 'Agios'],
            'Epargne': ['Placements', 'Assurance vie', 'Livrets']
        }

        self.motifs = {
            'GAB': ['Débit non effectué', 'Carte avalée', 'Code PIN bloqué', 'Montant incorrect'],
            'Carte bancaire': ['Paiement refusé', 'Opposition tardive', 'Débit frauduleux', 'Double prélèvement'],
            'TPE': ['Transaction non aboutie', 'Double débit', 'Montant erroné', 'Ticket non imprimé'],
            'Crédit personnel': ['Taux incorrect', 'Échéance manquante', 'Remboursement anticipé', 'Assurance'],
            'Crédit immobilier': ['Frais de dossier', 'Garantie', 'Taux variable', 'Report échéance'],
            'Crédit consommation': ['Taux erroné', 'Mensualité incorrecte', 'Clôture compte', 'Frais cachés'],
            'Tenue de compte': ['Prélèvement indû', 'Frais non justifiés', 'Double facturation', 'Tarif incorrect'],
            'Commissions': ['Commission non prévue', 'Taux abusif', 'Facturation erronée', 'Virement international'],
            'Agios': ['Calcul incorrect', 'Date valeur erronée', 'Taux non respecté', 'Dépassement autorisé'],
            'Placements': ['Rendement non conforme', 'Frais cachés', 'Information erronée', 'Rachat retardé'],
            'Assurance vie': ['Rachat différé', 'Arbitrage non effectué', 'Frais de gestion', 'Bénéficiaire'],
            'Livrets': ['Rémunération incorrecte', 'Plafond dépassé', 'Blocage indû', 'Clôture']
        }

        # Taux de fondement par catégorie (2024)
        self.success_rates_2024 = {
            'GAB': 0.75,
            'Carte bancaire': 0.68,
            'TPE': 0.72,
            'Crédit personnel': 0.55,
            'Crédit immobilier': 0.48,
            'Crédit consommation': 0.62,
            'Tenue de compte': 0.42,
            'Commissions': 0.38,
            'Agios': 0.52,
            'Placements': 0.35,
            'Assurance vie': 0.45,
            'Livrets': 0.50
        }

        # Drift 2025: légère dégradation du taux de fondement (banque plus stricte)
        self.success_rates_2025 = {k: v * 0.93 for k, v in self.success_rates_2024.items()}

    def generate_dataset(self, n_samples=33000, year=2024, start_date='2024-01-01'):
        """
        Génère un dataset de réclamations

        Args:
            n_samples: Nombre de réclamations
            year: Année (2024 ou 2025)
            start_date: Date de début
        """
        success_rates = self.success_rates_2024 if year == 2024 else self.success_rates_2025

        # Ajustement de distribution pour drift temporel
        drift_factor = 1.0 if year == 2024 else 1.15  # Plus de réclamations complexes en 2025

        data = []
        start = pd.to_datetime(start_date)

        for i in range(n_samples):
            # Date aléatoire dans l'année
            days_offset = np.random.randint(0, 365)
            date_qualification = start + timedelta(days=days_offset)

            # Sélection famille et catégorie
            famille = np.random.choice(list(self.familles.keys()))
            categorie = np.random.choice(self.familles[famille])
            motif = np.random.choice(self.motifs[categorie])

            # Fondement basé sur probabilités + bruit
            base_prob = success_rates.get(categorie, 0.5)

            # Ajout de facteurs influençant le fondement
            montant_factor = np.random.normal(0, 0.05)  # Montant élevé = plus fondé
            anciennete_factor = np.random.normal(0, 0.03)  # Ancienneté = plus fondé

            final_prob = np.clip(base_prob + montant_factor + anciennete_factor, 0.1, 0.9)
            fondee = 1 if np.random.random() < final_prob else 0

            # Montant demandé (distribution réaliste avec drift)
            if famille == 'Monétique':
                montant = np.random.lognormal(5.3, 1.3) * drift_factor  # Médiane ~200€
            elif famille == 'Crédit':
                montant = np.random.lognormal(7.8, 1.6) * drift_factor  # Médiane ~2500€
            elif famille == 'Frais bancaires':
                montant = np.random.lognormal(3.2, 0.9) * drift_factor  # Médiane ~25€
            else:  # Epargne
                montant = np.random.lognormal(6.8, 1.9) * drift_factor  # Médiane ~900€

            montant = round(montant, 2)

            # Client et caractéristiques
            client_id = f"CLI_{np.random.randint(10000, 99999)}"
            anciennete = max(0.1, np.random.exponential(4.2))  # Moyenne 4.2 ans

            # PNB (corrélé au montant et à l'ancienneté)
            pnb_base = montant * np.random.uniform(8, 60) * (1 + anciennete / 10)
            pnb = max(100, pnb_base + np.random.normal(0, pnb_base * 0.25))
            pnb = round(pnb, 2)

            # Banque privée (corrélé au PNB)
            prob_bp = 0.05 if pnb < 10000 else (0.25 if pnb < 50000 else 0.60)
            banque_privee = 'OUI' if np.random.random() < prob_bp else 'NON'

            # Segment (basé sur PNB)
            if pnb > 50000:
                segment = 'Premium'
            elif pnb > 15000:
                segment = 'Particuliers'
            else:
                segment = 'Grand Public'

            # Canal de réclamation
            canal_weights = [0.45, 0.30, 0.15, 0.10] if year == 2024 else [0.35, 0.30, 0.20, 0.15]  # Drift: plus digital
            canal = np.random.choice(['Agence', 'Téléphone', 'Email', 'Application mobile'], p=canal_weights)

            # Délai de traitement (jours)
            delai_base = {'Agence': 8, 'Téléphone': 12, 'Email': 15, 'Application mobile': 10}
            delai = max(1, int(np.random.normal(delai_base[canal], 4)))

            # Age du client
            age = int(np.clip(np.random.normal(45, 15), 18, 85))

            # Nombre de produits détenus
            nb_produits = int(np.clip(np.random.poisson(2.5), 1, 10))

            # Réclamations précédentes (Poisson)
            nb_reclamations_precedentes = int(np.random.poisson(0.8))

            data.append({
                'No_Demande': f'REC_{year}_{i+1:06d}',
                'Date_de_Qualification': date_qualification.strftime('%Y-%m-%d'),
                'Famille_Produit': famille,
                'Categorie': categorie,
                'Motif_Reclamation': motif,
                'Montant_demande': montant,
                'PNB_cumule': pnb,
                'ID_Client': client_id,
                'Anciennete_annees': round(anciennete, 2),
                'Banque_Privee': banque_privee,
                'Segment': segment,
                'Canal_Reclamation': canal,
                'Delai_traitement_jours': delai,
                'Age_client': age,
                'Nb_produits': nb_produits,
                'Nb_reclamations_precedentes': nb_reclamations_precedentes,
                'Fondee': fondee  # Variable cible
            })

        df = pd.DataFrame(data)
        return df

    def save_datasets(self, output_dir='ml_pipeline/data/raw'):
        """Génère et sauvegarde les datasets 2024 et 2025"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🔄 Génération des données 2024...")
        df_2024 = self.generate_dataset(n_samples=33000, year=2024, start_date='2024-01-01')
        path_2024 = output_path / 'reclamations_2024.xlsx'
        df_2024.to_excel(path_2024, index=False)
        print(f"✅ Données 2024 sauvegardées: {path_2024}")
        print(f"   - {len(df_2024)} réclamations")
        print(f"   - Taux fondées: {df_2024['Fondee'].mean():.1%}")
        print(f"   - Montant moyen: {df_2024['Montant_demande'].mean():.2f}€")

        print("\n🔄 Génération des données 2025 (avec drift)...")
        df_2025 = self.generate_dataset(n_samples=8000, year=2025, start_date='2025-01-01')
        path_2025 = output_path / 'reclamations_2025.xlsx'
        df_2025.to_excel(path_2025, index=False)
        print(f"✅ Données 2025 sauvegardées: {path_2025}")
        print(f"   - {len(df_2025)} réclamations")
        print(f"   - Taux fondées: {df_2025['Fondee'].mean():.1%} (drift: {((df_2025['Fondee'].mean() / df_2024['Fondee'].mean()) - 1) * 100:+.1f}%)")
        print(f"   - Montant moyen: {df_2025['Montant_demande'].mean():.2f}€ (drift: {((df_2025['Montant_demande'].mean() / df_2024['Montant_demande'].mean()) - 1) * 100:+.1f}%)")

        print("\n📊 Statistiques comparatives:")
        print("\nRépartition par famille (2024):")
        print(df_2024['Famille_Produit'].value_counts())

        print("\nRépartition par famille (2025):")
        print(df_2025['Famille_Produit'].value_counts())

        return df_2024, df_2025


if __name__ == "__main__":
    generator = ReclamationDataGenerator(seed=42)
    df_2024, df_2025 = generator.save_datasets()
    print("\n✅ Génération terminée !")
