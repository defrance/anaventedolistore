import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Configuration pour des graphiques plus jolis
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyser_ventes_francophones(fichier_csv, colonne_pays='pays', colonne_date='date', regroupement='M', date_debut='2020-01-01'):
    """
    Analyse les ventes par pays et génère un graphique comparant 
    les ventes francophones vs autres pays par période
    
    Parameters:
    fichier_csv (str): Chemin vers le fichier CSV
    colonne_pays (str): Nom de la colonne contenant les pays
    colonne_date (str): Nom de la colonne contenant les dates
    regroupement (str): Type de regroupement temporel
                       'W' = semaine, 'M' = mois, 'Q' = trimestre, 'Y' = année
    date_debut (str): Date de début d'analyse au format 'YYYY-MM-DD' (par défaut '2020-01-01')
    """
    
    # Lecture du fichier CSV
    try:
        # Lecture avec séparateur point-virgule pour les fichiers Dolibarr
        df = pd.read_csv(fichier_csv, sep=';')
        print(f"Fichier chargé avec succès: {len(df)} lignes")
    except FileNotFoundError:
        print(f"Erreur: Le fichier {fichier_csv} n'a pas été trouvé")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        return
    
    # Vérification des colonnes     
    if colonne_pays not in df.columns:
        print(f"Erreur: La colonne '{colonne_pays}' n'existe pas dans le fichier")
        print(f"Colonnes disponibles: {list(df.columns)}")
        return
    
    if colonne_date not in df.columns:
        print(f"Erreur: La colonne '{colonne_date}' n'existe pas dans le fichier")
        print(f"Colonnes disponibles: {list(df.columns)}")
        return
    
    # Conversion de la colonne date
    try:
        df[colonne_date] = pd.to_datetime(df[colonne_date])
    except Exception as e:
        print(f"Erreur lors de la conversion des dates: {e}")
        return
    
    # Filtrage des données selon la date de début spécifiée
    try:
        date_limite = pd.to_datetime(date_debut)
        df_original_count = len(df)
        df = df[df[colonne_date] >= date_limite]
        print(f"Filtrage des données après le {date_debut}: {len(df)} lignes conservées sur {df_original_count}")
        
        if len(df) == 0:
            print(f"Aucune donnée trouvée après le {date_debut}")
            return
    except Exception as e:
        print(f"Erreur lors du filtrage par date (format attendu: YYYY-MM-DD): {e}")
        return
    
    # Définition des codes pays francophones (ISO 2 lettres)
    pays_francophones = ['fr', 'be', 'ch', 'nc', 'ca', 'lu']  # France, Belgique, Suisse, Nouvelle-Calédonie, Canada, Luxembourg

    # Normalisation des codes pays (majuscules vers minuscules, suppression espaces)
    df[colonne_pays] = df[colonne_pays].str.lower().str.strip()
    
    # Classification francophone/non-francophone
    df['est_francophone'] = df[colonne_pays].isin(pays_francophones)
    
    # Définition des libellés pour les périodes
    regroupements_labels = {
        'W': 'semaine',
        'M': 'mois', 
        'Q': 'trimestre',
        'Y': 'année'
    }
    
    if regroupement not in regroupements_labels:
        print(f"Erreur: regroupement '{regroupement}' non supporté. Utilisez: W (semaine), M (mois), Q (trimestre), Y (année)")
        return
    
    periode_label = regroupements_labels[regroupement]
    
    # Extraction de la période selon le regroupement choisi
    df['periode'] = df[colonne_date].dt.to_period(regroupement)
    
    # Comptage des ventes par période et type
    ventes_par_periode = df.groupby(['periode', 'est_francophone']).size().unstack(fill_value=0)
    
    # Renommage des colonnes
    ventes_par_periode.columns = ['Autres pays', 'Pays francophones']
    
    # Si une colonne manque, on l'ajoute avec des zéros
    if 'Autres pays' not in ventes_par_periode.columns:
        ventes_par_periode['Autres pays'] = 0
    if 'Pays francophones' not in ventes_par_periode.columns:
        ventes_par_periode['Pays francophones'] = 0   
    
    # Calcul du total pour chaque période
    ventes_par_periode['Total'] = ventes_par_periode['Autres pays'] + ventes_par_periode['Pays francophones']
    
    # Calcul des pourcentages
    ventes_pourcentages = ventes_par_periode.copy()
    # Éviter la division par zéro
    mask_non_zero = ventes_par_periode['Total'] > 0
    ventes_pourcentages.loc[mask_non_zero, 'Pays francophones'] = (
        ventes_par_periode.loc[mask_non_zero, 'Pays francophones'] / 
        ventes_par_periode.loc[mask_non_zero, 'Total'] * 100
    )
    ventes_pourcentages.loc[mask_non_zero, 'Autres pays'] = (
        ventes_par_periode.loc[mask_non_zero, 'Autres pays'] / 
        ventes_par_periode.loc[mask_non_zero, 'Total'] * 100
    )
    # Le total est toujours 100% quand il y a des ventes
    ventes_pourcentages.loc[mask_non_zero, 'Total'] = 100
    ventes_pourcentages.loc[~mask_non_zero, :] = 0
    
    # Création du graphique avec double axe Y
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Conversion de l'index en datetime pour un meilleur affichage
    x_dates = ventes_pourcentages.index.to_timestamp()
    
    # Premier axe Y : Pourcentages
    line1 = ax1.plot(x_dates, ventes_pourcentages['Pays francophones'], 
            marker='o', linewidth=2.5, label='Pays francophones (%)', color='#2E86AB')
    line2 = ax1.plot(x_dates, ventes_pourcentages['Autres pays'], 
            marker='s', linewidth=2.5, label='Autres pays (%)', color='#A23B72')
    
    # Ajout des valeurs sur les points pour les pourcentages
    for i, (date, franco, autres) in enumerate(zip(x_dates, ventes_pourcentages['Pays francophones'], ventes_pourcentages['Autres pays'])):
        # Valeurs pour pays francophones (en haut)
        ax1.annotate(f'{franco:.1f}%', 
                    (date, franco), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=9, 
                    color='#2E86AB',
                    fontweight='bold')
        
        # Valeurs pour autres pays (en bas)
        ax1.annotate(f'{autres:.1f}%', 
                    (date, autres), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center', 
                    fontsize=9, 
                    color='#A23B72',
                    fontweight='bold')
    
    # Configuration du premier axe Y (pourcentages)
    ax1.set_xlabel(f'{periode_label.capitalize()}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pourcentage des ventes (%)', fontsize=12, fontweight='bold', color='black')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Création du second axe Y pour le nombre total de ventes
    ax2 = ax1.twinx()
    line3 = ax2.plot(x_dates, ventes_par_periode['Total'], 
            marker='^', linewidth=2.5, label='Nombre total de ventes', 
            color='#F18F01', linestyle='--', alpha=0.8)
    
    # Ajout des valeurs sur les points pour le nombre total
    for i, (date, total) in enumerate(zip(x_dates, ventes_par_periode['Total'])):
        ax2.annotate(f'{total}', 
                    (date, total), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=9, 
                    color='#F18F01',
                    fontweight='bold')
    
    # Configuration du second axe Y (nombres absolus)
    ax2.set_ylabel('Nombre total de ventes', fontsize=12, fontweight='bold', color='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    
    # Titre du graphique
    ax1.set_title(f'Évolution des ventes par {periode_label}\nPourcentages francophones/autres + Volume total', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Légendes combinées
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    # Ajustement de la mise en page
    plt.tight_layout()
    
    # Affichage du graphique
    plt.show()
    
    # Affichage des statistiques
    print("\n" + "="*50)
    print("RÉSUMÉ DES VENTES")
    print("="*50)
    
    total_francophones = ventes_par_periode['Pays francophones'].sum()
    total_autres = ventes_par_periode['Autres pays'].sum()
    total_general = total_francophones + total_autres
    
    print(f"Total ventes pays francophones: {total_francophones}")
    print(f"Total ventes autres pays: {total_autres}")
    print(f"Total général: {total_general}")
    
    if total_general > 0:
        pourcentage_franco = (total_francophones / total_general) * 100
        print(f"Pourcentage francophones: {pourcentage_franco:.1f}%")
        print(f"Pourcentage autres pays: {100-pourcentage_franco:.1f}%")
    
    print(f"\nDétail par {periode_label} (en pourcentage):")
    print(ventes_pourcentages.round(1).to_string())
    
    print(f"\nDétail par {periode_label} (nombres absolus):")
    print(ventes_par_periode.to_string())
    
    return ventes_par_periode, ventes_pourcentages

def analyser_multiple_regroupements(fichier_csv, colonne_pays='Customer country code', colonne_date='Sale Date', date_debut='2020-01-01'):
    """
    Analyse les ventes avec plusieurs regroupements temporels et affiche les résultats
    
    Parameters:
    fichier_csv (str): Chemin vers le fichier CSV
    colonne_pays (str): Nom de la colonne contenant les pays
    colonne_date (str): Nom de la colonne contenant les dates
    date_debut (str): Date de début d'analyse au format 'YYYY-MM-DD'
    """
    regroupements = {
        'W': 'Semaine',
        'M': 'Mois', 
        'Q': 'Trimestre',
        'Y': 'Année'
    }
    
    print("=== ANALYSE AVEC DIFFÉRENTS REGROUPEMENTS ===\n")
    
    for code, nom in regroupements.items():
        print(f"\n{'='*60}")
        print(f"ANALYSE PAR {nom.upper()}")
        print(f"{'='*60}")
        
        try:
            analyser_ventes_francophones(fichier_csv, colonne_pays, colonne_date, code, date_debut)
        except Exception as e:
            print(f"Erreur lors de l'analyse par {nom.lower()}: {e}")
        
        print("\n")

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez 'votre_fichier.csv' par le chemin vers votre fichier
    # Et ajustez les noms des colonnes si nécessaire
    
    fichier = "ventesdolibarr.csv"  # À remplacer par votre fichier
    
    # Choix du regroupement temporel :
    # 'W' = par semaine
    # 'M' = par mois (par défaut)
    # 'Q' = par trimestre  
    # 'Y' = par année
    regroupement_choisi = 'Q'  # Modifiez ici pour changer le regroupement
    
    # Date de début d'analyse (format YYYY-MM-DD)
    date_debut_analyse = '2014-01-01'  # Modifiez ici pour changer la date de début
    
    # Exemples d'autres dates de début possibles :
    # date_debut_analyse = '2021-01-01'  # Analyse depuis 2021
    # date_debut_analyse = '2023-01-01'  # Analyse des 2 dernières années
    # date_debut_analyse = '2024-01-01'  # Analyse de l'année en cours
    # date_debut_analyse = '2024-07-01'  # Analyse des 6 derniers mois
    
    # Si vos colonnes ont des noms différents, modifiez ici:
    ventes_data = analyser_ventes_francophones(
        fichier, 
        colonne_pays='Customer country code', 
        colonne_date='Sale Date',
        regroupement=regroupement_choisi,
        date_debut=date_debut_analyse
    )
    
    # Exemple d'utilisation avec la fonction d'analyse multiple :
    # analyser_multiple_regroupements(fichier, date_debut='2023-01-01')