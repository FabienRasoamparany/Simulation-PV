import fluxsolairefonctions as f
import matplotlib.pyplot as plt
import numpy as np
import traitementKT as kt
import math
import regressionpolyordre2 as p
import calculpaybackbenefice as eco
import pandas as pd

def main(excel_path):
    print("="*60 + "\n")
    print('Outil de simulation photovoltaique et projection économique')
    print("="*60 + "\n")
    print("-"*170 + "\n")
    print("ATTENTION!! Ce code calcul l'irradiation solaire en tenant comptes de tous les paramètres que vous pourrez donner. ""\n"+"Cependant il ne prend pas en compte la présence de masque. ""\n"+"Si on a la présence d'un arbre ou d'un autre masque solaire proche impactant, les resultats obetnus peuvent etre assez éloignés de la realité.""\n"+"Les calculs sont fiables sur une année ou mois mais de part l'imprevisiblité de la nebulosité il est illusoire de vouloir prédier une production à une heure précise")
    print(" "*120 + "\n")
    print("READ ME !! Si vous ignorez comment foncitonne ce code veuillez consulter la notice d'utilisation disponible à T:/pat-Maintenance/02-POLE_MAINTENANCE_SECURITE_ENERGIE/13-ELECTRICITE/8-PHOTOVOLTAÏQUE/00 - OUTILS SIMULATION SOLAIRE ")
    print("-"*170 + "\n")
    # Demande des paramètres d'entrée avec mise en forme et unités
    inclinaison = float(input('Veuillez entrer l\'inclinaison des panneaux photovoltaïques (en degrés) : '))
    albedo = 0.2
    latitude = 43.6  # ° Toulouse
    azimut_surface = float(input('Orientation des panneaux (en degrés) : '))
    rendement = float(input('Rendement des panneaux solaires (en %) : ')) - 2
    rendement = rendement / 100  # Conversion en fraction
    surface = float(input("Surface de la toiture exploitable (en m²) : "))
    type_toit = str(input("Type de toit choisir entre, 'bac acier','terrasse', 'tuile' ou 'ombrière' :"))
    duree = int(input('Durée du contrat d\'achat d\'électricité (en années) : '))
    prix = float(input('Prix d\'achat de l\'électricité définie dans le contrat (en €/kWh) : '))
    
    if type_toit =="bac acier":
        surface = surface*0.65
    if type_toit =="terrasse":
        surface = surface*0.5
    if type_toit=="tuile":
        surface = surface*0.35
    if type_toit == "ombrière":
        surface = surface * 0.9

    # Affichage des paramètres d'entrée
    print("\n" + "="*40)
    print("Résumé des paramètres d'entrée :")
    print(f"Inclinaison des panneaux      : {inclinaison}°")
    print(f"Orientation des panneaux      : {azimut_surface}°")
    print(f"Latitude du site (Toulouse)   : {latitude}°")
    print(f"Rendement des panneaux        : {rendement * 100+2}%")
    print(f"Surface de l'installation     : {surface} m²")
    print(f"Durée du contrat d'achat      : {duree} ans")
    print(f"Prix d'achat de l'électricité : {prix} €/kWh")
    print("="*40 + "\n")
    
    #----------------- REGRESSION POLYNOMIALE DU PRIX DES INSTALLATIONS PV -----------------#
    # Demande de paramètre par défaut ou personnalisé
    ask = str(input("Simulation économique, voulez-vous les paramètres par défaut ou personnalisés ? (defaut/perso) :")).lower()
    
    x_points = []
    y_points = []
    
    # Si l'utilisateur choisit les paramètres par défaut
    if ask == "defaut": 
        # Charger le fichier CSV en spécifiant l'encodage correct
        df = pd.read_csv(excel_path, sep=';', encoding='ISO-8859-1', header=None)
    
        # Nettoyer les colonnes en supprimant les espaces superflus
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # Nettoyer les valeurs des cellules
    
        # Remplacer les virgules par des points pour les nombres
        for col in df.columns[1:]:
            if df[col].dtype == 'object':  # Vérifier si la colonne est de type 'object'
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
        # Extraction des colonnes B (index 1) et C (index 2) et suppression des lignes contenant des valeurs nulles
        colonnes_B_C = df.iloc[:, [1, 2]].dropna()
    
        # Extraire la colonne B et C sous forme de listes
        liste_colonne_B = colonnes_B_C.iloc[:, 0].tolist()
        liste_colonne_C = colonnes_B_C.iloc[:, 1].tolist()
    
        # Ajouter toutes les valeurs dans x_points et y_points sans limitation à 7
        x_points = liste_colonne_B
        y_points = liste_colonne_C
    
    
    # Si l'utilisateur choisit d'entrer des paramètres personnalisés
    elif ask == "perso":    
        n = int(input("Combien de points voulez-vous entrer ? : "))
    
        # Collecte des points de surface et prix d'installation
        for i in range(n):
            x = float(input(f"Entrez la surface S{i+1} (en m²) : "))
            y = float(input(f"Entrez le prix P{i+1} (en euros) : "))
            x_points.append(x)
            y_points.append(y)
    
    else:
        print("Option non reconnue, veuillez entrer 'defaut' ou 'perso'.")
    
    # Effectuer la régression polynomiale d'ordre 2
    polynomial = p.polynomial_regression(x_points, y_points)
    
    # Afficher la fonction polynomiale
    
    #print(f"La fonction polynomiale calculée est : \n{polynomial}")
    
    
    # Demander à l'utilisateur une nouvelle valeur de surface pour évaluer le polynôme
    S_instal = surface
    Prix_instal = polynomial(S_instal)
    
    # Affichage du coût estimé de l'installation
    print("\n" + "="*90)
    print(f"\nPour une surface de panneaux effective de {S_instal} m², le coût estimé de l'installation est de {Prix_instal:,.0f} euros.\n".replace(',',' '))
    print("="*90)
    
    # Tracer le polynôme et les points d'origine
    p.plot_polynomial_with_point(polynomial, x_points, y_points,S_instal)
    
    #--------------------------------------------------#
    #--------------SURFACE HORIZONTALE-----------------#
    #--------------------------------------------------#
    
    #----------------Calcul H0 moyen------------------#
    H0_moyen=f.H0_moyen()
    #for i in range(len(H0_moyen)):
    #    print('Puissance recu par jour sans atm en moyenne au mois Shorizontale :',i+1,H0_moyen[i],'Wh/m²/j')
    
    #----------------Calcul Kt moyen------------------#
    Dh_sur_Gh=kt.DhsurGh()
    Vect_Dh_sur_Gh=Dh_sur_Gh['Kd']
    #print('Dh sur Gh par mois lissé sur 20 ans de mesures : \n',Dh_sur_Gh)
    #print('Juste les valeurs Dh sur Gh pour traitement \n',Vect_Dh_sur_Gh)
    # Coefficients du polynôme (ax^3 + bx^2 + cx + d)
    coefficients = [-3.108,5.531 ,-4.027 ,1.390 ]
    coefficients_tableaux = np.array([coefficients]*len(Dh_sur_Gh)) 
    #print('coefficients tableaux',coefficients_tableaux)
    for i in range(0,12):
        coefficients_tableaux[i,3]=coefficients_tableaux[i,3]-Vect_Dh_sur_Gh[i]
    
    #print('nouveau coefficient tableau : \n',coefficients_tableaux)
    
    Kt=[None]*12
    for i in range(0,12):
        Kt[i]=f.newton_raphson(coefficients_tableaux[i,:],0)
        Kt[i]=float(Kt[i])
    #print('Kt : \n',Kt)
    
    #----------------Calcul Gh moyen------------------#
    Gh_moyen=[None]*12
    for i in range(len(Gh_moyen)):
        #print(i)
        #print('KT et H0',Kt[i],H0_moyen[i])
        Gh_moyen[i]=Kt[i]*H0_moyen[i]
    
    #for i in range(0,12): 
    #    print('Puissance recu par jour en moyenne au mois Shorizontale :',i+1,'mois',Gh_moyen[i],'Wh/m²/j')
    
    #----------------Calcul Dh moyen------------------#
    Dh_moyen=[None]*len(Gh_moyen)
    for i in range(len(Gh_moyen)):
        Dh_moyen[i]=Gh_moyen[i]*(1.390-4.027*Kt[i]+5.531*Kt[i]**2-3.108*Kt[i]**3)
        Dh_moyen[i]=float(Dh_moyen[i])
        #print('Puissance recu rayonnement diffus horizontal en moyenne au mois :',i+1,'mois',Dh_moyen[i],'Wh/m²/j')
    
    #----------------Calcul Sh moyen------------------#
    Sh_moyen=[None]*len(Gh_moyen)
    for i in range(len(Gh_moyen)):
        Sh_moyen[i]=Gh_moyen[i]-Dh_moyen[i]
        #print('Puissance recu rayonnement direct horizontal en moyenne au mois :',i+1,'mois',Sh_moyen[i],'Wh/m²/j')
    #-----------------Calcul Gh, Sh, Dh par mois--------#
    jours_annee = [31,28.5,31,30,31,30,31,31,30,31,30,31]
    Gh_moyen_mois = []
    Dh_moyen_mois = []
    Sh_moyen_mois = []
    j=0
    for elem in jours_annee :
        Gh_moyen_mois.append(Gh_moyen[j]*elem/1000)
        Dh_moyen_mois.append(Dh_moyen[j]*elem/1000)
        Sh_moyen_mois.append(Sh_moyen[j]*elem/1000)
        j = j+1
    Eh_moyen_an = 0
    for i in range(len(Gh_moyen_mois)):
        Eh_moyen_an = Eh_moyen_an + Gh_moyen_mois[i]
    
    #print('Energie collectée par 1m² de panneaux horizontaux sur l année :',Eh_moyen_an, "kWh / m² /an")
    
    #--------------------------------------------------#
    #----------------SURFACE INCLINEE------------------#
    #--------------------------------------------------#
    coeff_rectif=[1.13,1.29,1.18,1.25,1.25,1.29,1.28,1.26,1.28,1.21,1.19,1.15]
    #----------------Calcul de Dh horaire pour chaque jour de l'année------------------#
    jours_moyen=[17, 16, 16, 15, 15, 11, 17, 16, 15, 15, 14, 10]
    jours_annee = [31,28.5,31,30,31,30,31,31,30,31,30,31]
    mois=np.arange(1,13)
    # Tableau pour stocker la somme des rayonnements moyens par mois
    rayonnement_moyen_par_mois = np.zeros(12)
    
    # Remplir le tableau pour chaque mois
    for mois_index in range(12):
        jour_ref = jours_moyen[mois_index]  # Le jour moyen de référence du mois
        Dh = Dh_moyen[mois_index]  # Rayonnement journalier moyen pour le mois
        nb_jours_mois = int(jours_annee[mois_index])  # Nombre de jours dans le mois
        
        # Calcul du rayonnement pour chaque jour de ce mois
        somme_rayonnement_mois = 0
        for jour in range(nb_jours_mois):
            radiation_jour = f.calcul_radiation_journalier_Dh(jour_ref, mois_index+1, Dh, latitude)
            somme_rayonnement_mois += np.sum(radiation_jour)  # Ajouter la somme du rayonnement journalier
    
        # Moyenne du rayonnement sur le mois
        moyenne_rayonnement_mois = somme_rayonnement_mois / (nb_jours_mois * 24)  # Moyenne par heure sur tout le mois
        rayonnement_moyen_par_mois[mois_index] = rendement* surface * moyenne_rayonnement_mois
    
    # Tableau pour stocker les résultats : 24 heures en ligne, et tous les jours de l'année en colonnes
    rayonnement_horaire_annee_D = np.zeros((24, 365))
    
    # Remplir le tableau pour chaque jour de l'année
    jour_courant = 0  # Indice du jour de l'année (0 à 364)
    for mois_index in range(12):
        jour_ref = jours_moyen[mois_index]  # Le jour moyen de référence du mois
        Dh = Dh_moyen[mois_index]  # Rayonnement journalier moyen pour le mois
        nb_jours_mois = int(jours_annee[mois_index])  # Nombre de jours dans le mois
        
        # Calcul du rayonnement pour chaque jour de ce mois
        for jour in range(nb_jours_mois):
            # Calcul du rayonnement pour chaque heure de la journée
            radiation_jour = f.calcul_radiation_journalier_Dh(jour_ref, mois_index+1, Dh, latitude)
            
            # Insérer les valeurs de radiation horaire dans le tableau pour chaque heure
            for heure in range(24):
                rayonnement_horaire_annee_D[heure, jour_courant] = radiation_jour[heure]
            
            jour_courant += 1  # Passer au jour suivant
    
    #----------------Calcul de Gh horaire pour chaque jour de l'année------------------#
    
    # Tableau pour stocker la somme des rayonnements moyens par mois
    rayonnement_moyen_par_mois = np.zeros(12)
    
    # Remplir le tableau pour chaque mois
    for mois_index in range(12):
        jour_ref = jours_moyen[mois_index]  # Le jour moyen de référence du mois
        Gh = Gh_moyen[mois_index]  # Rayonnement journalier moyen pour le mois
        nb_jours_mois = int(jours_annee[mois_index])  # Nombre de jours dans le mois
        
        # Calcul du rayonnement pour chaque jour de ce mois
        somme_rayonnement_mois = 0
        for jour in range(nb_jours_mois):
            radiation_jour = f.calcul_radiation_journalier_Gh(jour_ref, mois_index + 1, Gh, latitude)
            somme_rayonnement_mois += np.sum(radiation_jour)  # Ajouter la somme du rayonnement journalier
    
        # Moyenne du rayonnement sur le mois
        moyenne_rayonnement_mois = somme_rayonnement_mois / ((nb_jours_mois * 24)* coeff_rectif[mois_index])  # Moyenne par heure sur tout le mois
        rayonnement_moyen_par_mois[mois_index] = moyenne_rayonnement_mois
    
    # Tableau pour stocker les résultats : 24 heures en ligne, et tous les jours de l'année en colonnes
    rayonnement_horaire_annee_G = np.zeros((24, int(np.sum(jours_annee))))
    
    # Remplir le tableau pour chaque jour de l'année
    jour_courant = 0  # Indice du jour de l'année (0 à 364)
    for mois_index in range(12):
        jour_ref = jours_moyen[mois_index]  # Le jour moyen de référence du mois
        Gh = Gh_moyen[mois_index]  # Rayonnement journalier moyen pour le mois
        nb_jours_mois = int(jours_annee[mois_index])  # Nombre de jours dans le mois
        
        # Calcul du rayonnement pour chaque jour de ce mois
        for jour in range(nb_jours_mois):
            # Calcul du rayonnement pour chaque heure de la journée
            radiation_jour = f.calcul_radiation_journalier_Gh(jour_ref, mois_index + 1, Gh, latitude)
            
            # Insérer les valeurs de radiation horaire dans le tableau pour chaque heure
            for heure in range(24):
                rayonnement_horaire_annee_G[heure, jour_courant] = radiation_jour[heure]
            
            jour_courant += 1  # Passer au jour suivant
    
    #----------------Calcul de Sh horaire pour chaque jour de l'année------------------#
    # Création du tableau pour le rayonnement horaire S
    rayonnement_horaire_annee_S = np.zeros_like(rayonnement_horaire_annee_G)
    
    # Calcul de rayonnement horaire S pour chaque jour de l'année
    for heure in range(24):
        rayonnement_horaire_annee_S[heure, :] = rayonnement_horaire_annee_G[heure, :] - rayonnement_horaire_annee_D[heure, :]
    
    #----------------Calcul de S horaire pour chaque jour de l'année------------------#
    # Initialisation des tableaux pour les résultats
    cos_teta_annee = np.zeros((24, 365))  # Tableau pour cosinus de l'angle d'incidence
    sin_elevation_annee = np.zeros((24, 365))  # Tableau pour sinus de l'élévation solaire
    S=np.zeros((24,365))
    jour_courant = 0  # Indice du jour de l'année (de 0 à 364)
    # Parcourir chaque mois
    for mois_index in range(12):
        jour_ref = jours_moyen[mois_index]  # Le jour moyen de référence du mois
        Dh = Sh_moyen[mois_index]  # Rayonnement journalier moyen pour le mois
        nb_jours_mois = int(jours_annee[mois_index])  # Nombre de jours dans le mois
    
        # Parcourir chaque jour du mois
        for jour in range(nb_jours_mois):
            jour_N = f.Jour_N(jour_ref, mois_index + 1)  # Numéro du jour dans l'année
            declinaison = f.declinaison_solaire(jour_N)  # Calcul de la déclinaison solaire
    
            # Parcourir chaque heure de la journée
            for heure in range(24):
                angle_horaire = f.angle_horaire(heure)  # Calcul de l'angle horaire solaire
                h = f.altitude_solaire(latitude, declinaison, angle_horaire)  # Élévation solaire
    
                # Calcul du sinus de l'élévation solaire
                sin_elevation = f.sinus_elevation(h)
                sin_elevation_annee[heure, jour_courant] = sin_elevation
    
                # Calcul de l'azimut solaire
                azimut = f.azimut_solaire(latitude, declinaison, angle_horaire)
    
                # Calcul du cosinus de l'angle d'incidence (cos_teta)
                cos_teta = f.cosinus_teta(h, azimut, azimut_surface, inclinaison)
                cos_teta_annee[heure, jour_courant] = cos_teta
    
                # Calcul de Sh (rayonnement horaire S)
                if sin_elevation > 0:  # Calcul uniquement si l'élévation est positive
                    S[heure,jour_courant]= (rayonnement_horaire_annee_S[heure, jour_courant] * cos_teta) / sin_elevation
                else:
                    S[heure,jour_courant] = 0  # Pas de rayonnement si l'élévation est négative (nuit)
                if S[heure,jour_courant] >=0:
                    continue
                else:
                    S[heure,jour_courant]=0
            jour_courant += 1  # Passer au jour suivant
    
    # Affichage des résultats
    
    # 3. Affichage graphique du rayonnement solaire horaire incliné (S) sur l'année
    plt.figure(figsize=(12, 6))
    plt.imshow(S, aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(label='Rayonnement solaire horaire direct sur surface incliné')
    plt.xlabel('Jour de l\'année')
    plt.ylabel('Heure de la journée')
    plt.title('Rayonnement solaire horaire incliné (S) sur l\'année')
    plt.show()
    
    # Calcul du rayonnement total moyen pour chaque mois
    rayonnement_moyen_mois_S = np.zeros(12)
    jour_courant = 0
    for mois_index in range(12):
        nb_jours_mois = int(jours_annee[mois_index])
        somme_rayonnement_mois = 0
    
        for jour in range(nb_jours_mois):
            somme_rayonnement_mois += np.sum(S[:, jour_courant])  # Somme du rayonnement horaire pour chaque jour du mois
            jour_courant += 1
    
        # Calcul de la moyenne pour le mois
        rayonnement_moyen_mois_S[mois_index] = rendement* surface * somme_rayonnement_mois / ((nb_jours_mois * 24)*coeff_rectif[mois_index])  # Moyenne sur toutes les heures
    
    #----------------Calcul de G horaire pour chaque jour de l'année------------------#
    G=np.zeros((24,365))
    G= S + (1/2+math.cos(math.radians(inclinaison))/2)*rayonnement_horaire_annee_D + albedo*rayonnement_horaire_annee_G*(1/2 - math.cos(math.radians(inclinaison))/2)
    
    #----------------Calcul de D horaire pour chaque jour de l'année------------------#
    D=np.zeros((24,365))
    D = G - S
    
    # Affichage de la carte de chaleur (heatmap)
    plt.figure(figsize=(12, 6))
    plt.imshow(D, aspect='auto', cmap='hot', origin='lower')
    
    # Configuration des labels
    plt.colorbar(label='Rayonnement total D (Wh/m²)')
    plt.xlabel('Jour de l\'année')
    plt.ylabel('Heure de la journée')
    plt.title('Rayonnement solaire horaire diffus sur surface incliné')
    plt.show()
    
    # Calcul du rayonnement total moyen pour chaque mois
    rayonnement_moyen_mois_D = np.zeros(12)
    jour_courant = 0
    for mois_index in range(12):
        nb_jours_mois = int(jours_annee[mois_index])
        somme_rayonnement_mois = 0
    
        for jour in range(nb_jours_mois):
            somme_rayonnement_mois += np.sum(D[:, jour_courant])  # Somme du rayonnement horaire pour chaque jour du mois
            jour_courant += 1
    
        # Calcul de la moyenne pour le mois
        rayonnement_moyen_mois_D[mois_index] = rendement* surface * somme_rayonnement_mois  / ((nb_jours_mois * 24)* coeff_rectif[mois_index])  # Moyenne sur toutes les heures
    
    #--------------------------------Affichage de G-----------------------------------#,
    # Affichage de la carte de chaleur (heatmap)
    plt.figure(figsize=(12, 6))
    plt.imshow(G, aspect='auto', cmap='hot', origin='lower')
    
    # Configuration des labels
    plt.colorbar(label='Rayonnement total G (Wh/m²)')
    plt.xlabel('Jour de l\'année')
    plt.ylabel('Heure de la journée')
    plt.title('Rayonnement solaire horaire total sur surface incliné')
    
    # Afficher le graphique
    plt.show()
    
    # Calcul du rayonnement total moyen pour chaque mois
    rayonnement_moyen_mois_G = np.zeros(12)
    jour_courant = 0
    Rayonnemnent_G_an = 0
    for mois_index in range(12):
        nb_jours_mois = int(jours_annee[mois_index])
        somme_rayonnement_mois = 0
    
        for jour in range(nb_jours_mois):
            somme_rayonnement_mois += np.sum(G[:, jour_courant])  # Somme du rayonnement horaire pour chaque jour du mois
            jour_courant += 1
        # Calcul de la moyenne pour le mois
        
        rayonnement_moyen_mois_G[mois_index] = rendement* surface * somme_rayonnement_mois  / ((nb_jours_mois * 24)* coeff_rectif[mois_index])  # Moyenne sur toutes les heures
      # Ajouter le rayonnement mensuel total au rayonnement annuel
    for e in rayonnement_moyen_mois_G:
        Rayonnemnent_G_an = Rayonnemnent_G_an + e
    
    bar_width = 0.25
    
    r1 = np.arange(len(rayonnement_moyen_mois_D))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    plt.figure(figsize=(10, 6))
    plt.bar(r1,rayonnement_moyen_mois_G , color='b', width=bar_width, edgecolor='grey', label='Gh (Total)')
    plt.bar(r2,rayonnement_moyen_mois_D , color='g', width=bar_width, edgecolor='grey', label='Dh (Diffus)')
    plt.bar(r3, rayonnement_moyen_mois_S, color='r', width=bar_width, edgecolor='grey', label='Sh (Directe)')
    
    plt.xlabel('Mois', fontweight='bold', fontsize=15)
    plt.ylabel('Production PV (kWh)', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width for r in range(len(rayonnement_moyen_mois_G))], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title("Production photovoltaique sur l'année", fontsize=18, fontweight='bold')
    
    plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    #---------------------------VERIFICATION DES RESULTATS-----------------------------------#
    #----------------Calcul Gh moyen (base de donnee pgvis------------------#
    #somme_rayonnement_Gh_pgvis=0
    #Gh_verif=kt.Gh()
    
    #---------------------------------------ANALYSE ECONOMIQUE------------------------------------#
    
    eco.calcul_payback_benefice_avec_entretien_baisse(Rayonnemnent_G_an,prix,duree,Prix_instal,type_toit)
    return
    
