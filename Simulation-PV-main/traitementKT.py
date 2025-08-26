import pandas as pd
import sklearn 
import tarfile
import csv 

def DhsurGh():
    kt_path="/content/Simulation-PV/Monthlydata_43.579_1.503_SA2_2005_2020.csv"
    Kt=pd.read_csv(kt_path, header=None, names=['data']) #lecture du fichier de données


    Kt[['year', 'month', 'Kd']] =Kt['data'].str.split('\t\t', expand=True)  #séparation en 3 colonnes 

    Kt=Kt.drop(columns=['data'])
    Kt['Kd']=pd.to_numeric(Kt['Kd'],errors='coerce')  

    Kt['Kd']=Kt['Kd'].astype(float)
    Kt=Kt.dropna()

    grouped=Kt.groupby('month')['Kd'].mean().reset_index() #matrice des Kt de chaque mois moyenné de 2005 à 2020
    #print('matrice groupée',grouped)

    # Création d'un dictionnaire pour mapper les noms des mois à leurs représentations numériques
    mois_en_chiffre = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12 
        }
    grouped_chiffres=grouped.copy()
    grouped_chiffres['month'] = grouped_chiffres['month'].map(mois_en_chiffre) #application du dictionnaire a tous les elements de grouped pour avoir les mois en numéros
    grouped_chiffres = grouped_chiffres.sort_values('month') #trie la colonne month par ordre croissant
    grouped_chiffres = grouped_chiffres.reset_index(drop=True)
    chiffres_en_mois = {
        1:'Jan',
        2:'Feb' ,
        3:'Mar',
        4:'Apr',
        5:'May',
        6:'Jun',
        7:'Jul',
        8:'Aug',
        9:'Sep',
        10:'Oct',
        11:'Nov',
        12:'Dec' 
        }
    grouped_chiffres['month']=grouped_chiffres['month'].map(chiffres_en_mois)

    return grouped_chiffres

def Gh():
    # Chemin du fichier CSV
    Gh_path = 'C:/Users/m.leclercq-stag/Downloads/MonthlydataGh_43.615_1.417_SA2_2005_2020.csv'
    
    # Lecture du fichier avec séparation par tabulation
    Gh = pd.read_csv(Gh_path, header=None, names=['data'])  # Lecture du fichier de données
    
    # Séparation en colonnes (une seule tabulation semble être correcte)
    Gh[['year', 'month', 'H(h)_m']] = Gh['data'].str.split('\t\t', expand=True)
    
    # Suppression de la colonne 'data'
    Gh = Gh.drop(columns=['data'])
    
    # Conversion de la colonne 'H(h)_m' en nombre
    Gh['H(h)_m'] = pd.to_numeric(Gh['H(h)_m'], errors='coerce')
    
    # Suppression des valeurs NaN
    Gh = Gh.dropna()

    # Mapping des abréviations des mois vers des chiffres
    mois_en_chiffre = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }
    
    # Conversion de la colonne 'month' à l'aide du mapping
    Gh['month'] = Gh['month'].map(mois_en_chiffre)
    
    # Calcul de la moyenne mensuelle du rayonnement
    grouped = Gh.groupby('month')['H(h)_m'].mean().reset_index()
    grouped = grouped.sort_values('month').reset_index(drop=True)
    # Mapping des mois numériques en mois abrégés (facultatif)
    chiffres_en_mois = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec'
    }
    
    grouped['month'] = grouped['month'].map(chiffres_en_mois)
    
    # Trier les mois et réinitialiser les index
    
    
    return grouped['H(h)_m']
