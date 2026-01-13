"""
Générateur de données réalistes avec les VRAIES colonnes de production
Colonnes 2024 et 2025 conformes aux données réelles de la banque
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class RealColumnDataGenerator:
    """Génère des données avec les colonnes exactes de production"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed

        # Métadonnées réalistes
        self.familles = {
            'Monétique': ['GAB', 'Carte bancaire', 'TPE'],
            'Crédit': ['Crédit personnel', 'Crédit immobilier', 'Crédit consommation'],
            'Frais bancaires': ['Tenue de compte', 'Commissions', 'Agios'],
            'Epargne': ['Placements', 'Assurance vie', 'Livrets']
        }

        self.sous_categories = {
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

        # Taux de fondement par catégorie
        self.success_rates_2024 = {
            'GAB': 0.75, 'Carte bancaire': 0.68, 'TPE': 0.72,
            'Crédit personnel': 0.55, 'Crédit immobilier': 0.48, 'Crédit consommation': 0.62,
            'Tenue de compte': 0.42, 'Commissions': 0.38, 'Agios': 0.52,
            'Placements': 0.35, 'Assurance vie': 0.45, 'Livrets': 0.50
        }

        # Drift 2025: légère dégradation
        self.success_rates_2025 = {k: v * 0.93 for k, v in self.success_rates_2024.items()}

        # Régions du Maroc
        self.regions = [
            'Casablanca-Settat', 'Rabat-Salé-Kénitra', 'Marrakech-Safi',
            'Fès-Meknès', 'Tanger-Tétouan-Al Hoceïma', 'Oriental',
            'Souss-Massa', 'Béni Mellal-Khénifra', 'Drâa-Tafilalet'
        ]

        # Réseaux
        self.reseaux = ['Réseau Commercial', 'Réseau Entreprises', 'Banque Privée', 'Agences']

        # Groupes
        self.groupes = ['Particuliers', 'Professionnels', 'Entreprises', 'Institutionnels']

        # Segments
        self.segments = ['Grand Public', 'Particuliers', 'Premium', 'VVIP']

        # Canaux de réception
        self.canaux = ['Agence', 'Téléphone', 'Email', 'Application mobile', 'Courrier', 'Réseaux sociaux']

    def generate_2024_dataset(self, n_samples=33000):
        """Génère le dataset 2024 avec les colonnes exactes"""
        data = []
        start_date = pd.to_datetime('2024-01-01')

        for i in range(n_samples):
            # Date aléatoire en 2024
            days_offset = np.random.randint(0, 365)
            date_qualification = start_date + timedelta(days=days_offset)
            date_ouverture = date_qualification - timedelta(days=np.random.randint(1, 30))

            # Sélection famille et catégorie
            famille = np.random.choice(list(self.familles.keys()))
            categorie = np.random.choice(self.familles[famille])
            sous_cat = np.random.choice(self.sous_categories[categorie])

            # Fondement
            base_prob = self.success_rates_2024.get(categorie, 0.5)
            fondee = 1 if np.random.random() < base_prob else 0

            # Montant demandé
            if famille == 'Monétique':
                montant = np.random.lognormal(5.3, 1.3)
            elif famille == 'Crédit':
                montant = np.random.lognormal(7.8, 1.6)
            elif famille == 'Frais bancaires':
                montant = np.random.lognormal(3.2, 0.9)
            else:
                montant = np.random.lognormal(6.8, 1.9)

            montant = round(montant, 2)
            montant_reponse = round(montant * np.random.uniform(0.6, 1.0), 2) if fondee else 0.0

            # Client
            client_id = f"{np.random.randint(100000, 999999)}"
            numero_compte = f"{np.random.randint(1000000000, 9999999999)}"
            anciennete = max(0.1, np.random.exponential(4.2))

            # PNB
            pnb_base = montant * np.random.uniform(8, 60) * (1 + anciennete / 10)
            pnb = max(100, pnb_base + np.random.normal(0, pnb_base * 0.25))

            # Banque privée
            prob_bp = 0.05 if pnb < 10000 else (0.25 if pnb < 50000 else 0.60)
            banque_privee = 'OUI' if np.random.random() < prob_bp else 'NON'

            # Segment
            if pnb > 50000:
                segment = 'Premium' if np.random.random() < 0.7 else 'VVIP'
            elif pnb > 15000:
                segment = 'Particuliers'
            else:
                segment = 'Grand Public'

            # Région et réseau
            region = np.random.choice(self.regions)
            reseau = np.random.choice(self.reseaux, p=[0.6, 0.2, 0.1, 0.1])
            groupe = np.random.choice(self.groupes, p=[0.6, 0.2, 0.15, 0.05])

            # Statut
            statut = 'Clôturée' if np.random.random() < 0.85 else 'En cours'

            # Canal
            canal_weights = [0.40, 0.25, 0.20, 0.10, 0.03, 0.02]
            canal = np.random.choice(self.canaux, p=canal_weights)

            # Délai estimé
            delai = int(np.clip(np.random.normal(12, 5), 3, 30))

            # Type demande
            type_demande = 'Réclamation' if np.random.random() < 0.9 else 'Requête'

            # PP/PM
            pp_pm = 'PP' if np.random.random() < 0.85 else 'PM'

            # Marché
            marche = 'Particuliers' if pp_pm == 'PP' else 'Entreprises'

            # Recevable
            recevable = 'OUI' if np.random.random() < 0.95 else 'NON'

            # Financière ou non
            financiere = 'OUI' if montant > 0 else 'NON'

            # Wafacash
            wafacash = 'OUI' if famille == 'Monétique' and np.random.random() < 0.15 else 'NON'

            # Code agence
            code_agence = f"AG{np.random.randint(100, 999)}"
            libelle_agence = f"Agence {region.split('-')[0]}"

            # Date debut relation
            dt_debrel = date_qualification - timedelta(days=int(anciennete * 365))

            # Source et BAS (spécifiques à 2024)
            source = np.random.choice(['SOFER', 'GESREC', 'PORTAL', 'MANUEL'])
            bas = f"BAS{np.random.randint(100, 999)}"

            data.append({
                'No Demande': f'REC_2024_{i+1:06d}',
                'Source': source,
                'Type Demande': type_demande,
                'Région': region,
                'Réseau': reseau,
                'Groupe': groupe,
                'Statut': statut,
                'Nom': f'Client_{client_id}',
                'N compte': numero_compte,
                'Ouvert': date_ouverture.strftime('%Y-%m-%d'),
                'Famille Produit': famille,
                'Catégorie': categorie,
                'Sous-catégorie': sous_cat,
                'Marché': marche,
                'PP/PM': pp_pm,
                'Canal de Réception': canal,
                'Délai Estimé (j)': delai,
                'Segment': segment,
                'Code Agence / CA Principal': code_agence,
                'Libellé Agence / CA Principal': libelle_agence,
                'Code Entité Source': code_agence,
                'Libellé Entité Source': libelle_agence,
                'Banque Privé': banque_privee,
                'Financière ou non': financiere,
                'Fondee': fondee,
                'Wafacash': wafacash,
                'Montant de réponse': montant_reponse,
                'Montant demandé': montant,
                'Priorité Client': 'Haute' if pnb > 50000 else ('Moyenne' if pnb > 15000 else 'Standard'),
                'Entité Resp': code_agence,
                'Motif d\'irrecevabilité': '' if recevable == 'OUI' else 'Hors périmètre',
                'Recevable': recevable,
                'Date de Qualification': date_qualification.strftime('%Y-%m-%d'),
                'BAS': bas,
                'Montant': montant,
                'numero_compte': numero_compte,
                'idtfcl': client_id,
                'PNB analytique (vision commerciale) cumulé': round(pnb, 2),
                'dt_debrel': dt_debrel.strftime('%Y-%m-%d'),
                'anciennete_annees': round(anciennete, 2)
            })

        return pd.DataFrame(data)

    def generate_2025_dataset(self, n_samples=8000):
        """Génère le dataset 2025 avec les colonnes exactes"""
        data = []
        start_date = pd.to_datetime('2025-01-01')
        drift_factor = 1.15  # Plus de réclamations complexes

        for i in range(n_samples):
            # Date aléatoire en 2025
            days_offset = np.random.randint(0, 365)
            date_qualification = start_date + timedelta(days=days_offset)
            date_ouverture = date_qualification - timedelta(days=np.random.randint(1, 30))

            # Sélection famille et catégorie
            famille = np.random.choice(list(self.familles.keys()))
            categorie = np.random.choice(self.familles[famille])
            sous_cat = np.random.choice(self.sous_categories[categorie])

            # Fondement (avec drift)
            base_prob = self.success_rates_2025.get(categorie, 0.5)
            fondee = 1 if np.random.random() < base_prob else 0

            # Montant demandé (avec drift)
            if famille == 'Monétique':
                montant = np.random.lognormal(5.3, 1.3) * drift_factor
            elif famille == 'Crédit':
                montant = np.random.lognormal(7.8, 1.6) * drift_factor
            elif famille == 'Frais bancaires':
                montant = np.random.lognormal(3.2, 0.9) * drift_factor
            else:
                montant = np.random.lognormal(6.8, 1.9) * drift_factor

            montant = round(montant, 2)
            montant_reponse = round(montant * np.random.uniform(0.6, 1.0), 2) if fondee else 0.0

            # Client
            client_id = f"{np.random.randint(100000, 999999)}"
            numero_compte = f"{np.random.randint(1000000000, 9999999999)}"
            anciennete = max(0.1, np.random.exponential(4.2))

            # PNB (avec drift)
            pnb_base = montant * np.random.uniform(8, 60) * (1 + anciennete / 10)
            pnb = max(100, pnb_base + np.random.normal(0, pnb_base * 0.25))

            # Autres champs (similaires à 2024)
            prob_bp = 0.05 if pnb < 10000 else (0.25 if pnb < 50000 else 0.60)
            banque_privee = 'OUI' if np.random.random() < prob_bp else 'NON'

            if pnb > 50000:
                segment = 'Premium' if np.random.random() < 0.7 else 'VVIP'
            elif pnb > 15000:
                segment = 'Particuliers'
            else:
                segment = 'Grand Public'

            region = np.random.choice(self.regions)
            reseau = np.random.choice(self.reseaux, p=[0.6, 0.2, 0.1, 0.1])
            groupe = np.random.choice(self.groupes, p=[0.6, 0.2, 0.15, 0.05])
            statut = 'Clôturée' if np.random.random() < 0.85 else 'En cours'
            canal_weights = [0.35, 0.25, 0.20, 0.15, 0.03, 0.02]  # Plus digital en 2025
            canal = np.random.choice(self.canaux, p=canal_weights)
            delai = int(np.clip(np.random.normal(12, 5), 3, 30))
            type_demande = 'Réclamation' if np.random.random() < 0.9 else 'Requête'
            pp_pm = 'PP' if np.random.random() < 0.85 else 'PM'
            marche = 'Particuliers' if pp_pm == 'PP' else 'Entreprises'
            recevable = 'OUI' if np.random.random() < 0.95 else 'NON'
            financiere = 'OUI' if montant > 0 else 'NON'
            wafacash = 'OUI' if famille == 'Monétique' and np.random.random() < 0.15 else 'NON'
            code_agence = f"AG{np.random.randint(100, 999)}"
            libelle_agence = f"Agence {region.split('-')[0]}"
            dt_debrel = date_qualification - timedelta(days=int(anciennete * 365))

            # Champs spécifiques à 2025
            demandeur = np.random.choice(['Titulaire', 'Mandataire', 'Héritier', 'Représentant légal'])
            code_gab = f"GAB{np.random.randint(1000, 9999)}" if famille == 'Monétique' else ''
            code_anomalie_gab = f"ERR{np.random.randint(100, 999)}" if famille == 'Monétique' else ''

            data.append({
                'No Demande': f'REC_2025_{i+1:06d}',
                'Type Demande': type_demande,
                'Région': region,
                'Réseau': reseau,
                'Groupe': groupe,
                'Statut': statut,
                'Nom': f'Client_{client_id}',
                'N compte': numero_compte,
                'Ouvert': date_ouverture.strftime('%Y-%m-%d'),
                'Famille Produit': famille,
                'Catégorie': categorie,
                'Sous-catégorie': sous_cat,
                'Marché': marche,
                'PP/PM': pp_pm,
                'Canal de Réception': canal,
                'Demandeur': demandeur,
                'Délai Estimé (j)': delai,
                'Segment': segment,
                'Code Agence / CA Principal': code_agence,
                'Libellé Agence / CA Principal': libelle_agence,
                'Code Entité Source': code_agence,
                'Libellé Entité Source': libelle_agence,
                'Banque Privé': banque_privee,
                'Financière ou non': financiere,
                'Fondee': fondee,
                'Wafacash': wafacash,
                'Montant de réponse': montant_reponse,
                'Montant demandé': montant,
                'Priorité Client': 'Haute' if pnb > 50000 else ('Moyenne' if pnb > 15000 else 'Standard'),
                'Entité Resp.': code_agence,
                'Motif d\'irrecevabilité': '' if recevable == 'OUI' else 'Hors périmètre',
                'Recevable': recevable,
                'Motif de rejet réponse UT': '',
                'Date Rejet réponse UT': '',
                'Motif de rejet UT': '',
                'Date Rejet UT': '',
                'Code anomalie GAB': code_anomalie_gab,
                'Code GAB': code_gab,
                'Motif dérogation': '',
                'Acteur dérogation': '',
                'Date de Qualification': date_qualification.strftime('%Y-%m-%d'),
                'Montant': montant,
                'numero_compte': numero_compte,
                'idtfcl': client_id,
                'PNB analytique (vision commerciale) cumulé': round(pnb, 2),
                'dt_debrel': dt_debrel.strftime('%Y-%m-%d'),
                'anciennete_annees': round(anciennete, 2)
            })

        return pd.DataFrame(data)

    def save_datasets(self, output_dir='ml_pipeline/data/raw'):
        """Génère et sauvegarde les datasets 2024 et 2025"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🔄 Génération des données 2024 (vraies colonnes)...")
        df_2024 = self.generate_2024_dataset(n_samples=33000)
        path_2024 = output_path / 'reclamations_2024.xlsx'
        df_2024.to_excel(path_2024, index=False)
        print(f"✅ Données 2024 sauvegardées: {path_2024}")
        print(f"   - {len(df_2024)} réclamations")
        print(f"   - {len(df_2024.columns)} colonnes")
        print(f"   - Taux fondées: {df_2024['Fondee'].mean():.1%}")
        print(f"   - Montant moyen: {df_2024['Montant demandé'].mean():.2f} MAD")

        print("\n🔄 Génération des données 2025 (vraies colonnes + drift)...")
        df_2025 = self.generate_2025_dataset(n_samples=8000)
        path_2025 = output_path / 'reclamations_2025.xlsx'
        df_2025.to_excel(path_2025, index=False)
        print(f"✅ Données 2025 sauvegardées: {path_2025}")
        print(f"   - {len(df_2025)} réclamations")
        print(f"   - {len(df_2025.columns)} colonnes")
        print(f"   - Taux fondées: {df_2025['Fondee'].mean():.1%} (drift: {((df_2025['Fondee'].mean() / df_2024['Fondee'].mean()) - 1) * 100:+.1f}%)")
        print(f"   - Montant moyen: {df_2025['Montant demandé'].mean():.2f} MAD (drift: {((df_2025['Montant demandé'].mean() / df_2024['Montant demandé'].mean()) - 1) * 100:+.1f}%)")

        print("\n📊 Colonnes 2024:")
        print(f"   {list(df_2024.columns)[:10]}...")

        print("\n📊 Colonnes 2025:")
        print(f"   {list(df_2025.columns)[:10]}...")

        return df_2024, df_2025


if __name__ == "__main__":
    generator = RealColumnDataGenerator(seed=42)
    df_2024, df_2025 = generator.save_datasets()
    print("\n✅ Génération terminée avec les vraies colonnes !")
