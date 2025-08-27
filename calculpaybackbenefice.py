def calcul_payback_benefice(production_annuelle_kWh, prix_achat_kWh, duree_contrat, investissement):
    # Calcul du revenu annuel
    revenu_annuel = production_annuelle_kWh * prix_achat_kWh
    
    # Calcul du temps de retour sur investissement (payback)
    if revenu_annuel > 0:
        temps_payback = investissement / revenu_annuel
    else:
        return "Le revenu annuel est nul ou négatif, impossible de calculer le temps de payback."
    
    # Calcul du bénéfice total sur la durée du contrat
    revenu_total_contrat = revenu_annuel * duree_contrat
    benefice_total = revenu_total_contrat - investissement
    
    # Affichage des résultats
    print(f"Temps de retour sur investissement (payback) : {temps_payback:.2f} années")
    print(f"Bénéfice total après {duree_contrat} ans : {benefice_total:.2f} €")
    
    return temps_payback, benefice_total

def calcul_payback_benefice_avec_entretien_baisse(production_annuelle_kWh, prix_achat_kWh, duree_contrat, investissement, type_install="toiture"):
    import numpy as np
    import matplotlib.pyplot as plt
    # Ajustement de l'investissement si ombrière
    if type_install == "ombrière":
        investissement *= 1.3  # Exemple : +30 % par rapport à une toiture
    if type_install == "ombrière":
        print("Entretien conseillé pour une ombrière (3% à 6%).")
    else:
        print("Entretien conseillé pour une toiture (2% à 5%).")
    entretien = int(input("Entretien (en % de l'investissement/an) : "))
    entretien_annuel = investissement * entretien/100  # Frais d'entretien annuel = 3% de l'investissement + inflation pour la maintenance
    production_actuelle = production_annuelle_kWh
    revenu_total_contrat = 0  # Initialisation des revenus totaux
    cout_total_entretien = 0  # Initialisation des coûts d'entretien
    temps_payback = None  # Initialisation du temps de payback
    production_an_perte=[]
    if type_install == "ombrière":
        production_baisse = 0.02  # 2% la première année
        perte_annuelle = 0.007  # 0.7% les années suivantes
    else:
        production_baisse = 0.015  # 1.5% la première année
        perte_annuelle = 0.005  # 0.5% les années suivantes


    for annee in range(1, duree_contrat + 1):
        # Calcul de la baisse de production annuelle
        if annee == 1:
            production_actuelle *= (1 - production_baisse )
        else:
            production_actuelle *= (1 - perte_annuelle)
        production_an_perte.append(production_actuelle)
        # Revenu annuel en fonction de la production réduite
        revenu_annuel = production_actuelle * prix_achat_kWh
        # Calcul du revenu total et des frais d'entretien
        revenu_total_contrat += revenu_annuel
        cout_total_entretien += entretien_annuel 
        entretien_annuel = entretien_annuel * 1.025 #inflation dans l'entretien
        
        # Vérification du temps de retour sur investissement (payback)
        if temps_payback is None and revenu_total_contrat - cout_total_entretien >= investissement:
            temps_payback = annee

    # Calcul du bénéfice total après le contrat
    benefice_total = revenu_total_contrat - cout_total_entretien - investissement

    duree_contrat = len(production_an_perte)  # Durée du contrat (nombre d'années)

    # Création des années pour l'axe des X
    annees = np.arange(1, duree_contrat + 1)

    # Définir la largeur des barres
    bar_width = 0.5

    # Création du graphique en barres
    plt.figure(figsize=(10, 6))
    plt.bar(annees, production_an_perte, color='b', width=bar_width, edgecolor='grey', label='Production elec')

    # Ajouter des labels et un titre
    plt.xlabel("Année d'exploitation", fontweight='bold', fontsize=15)
    plt.ylabel('Production (MWh)', fontweight='bold', fontsize=15)
    plt.title("Production installation PV sur la durée d'exploitation", fontsize=18, fontweight='bold')

    plt.xticks(annees)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajuster la mise en page
    plt.tight_layout()

    # Afficher le graphique
    prod = production_an_perte[len(production_an_perte)-1]
    plt.show()
    # Affichage des résultats économiques
    print("\n" + "="*70)
    print("Résumé des résultats économiques :")
    print("="*70)

    # Affichage du temps de retour sur investissement
    if temps_payback:
        print(f"Temps de retour sur investissement avec entretien (payback) : {temps_payback:.2f} années")
    else:
        print("Le temps de payback dépasse la durée du contrat.")

    print(f"Bénéfice total avec entretien après {duree_contrat} ans : {benefice_total:,.0f} €".replace(',', ' '))

    # Affichage du coût total de l'entretien
    print(f"Coût total de l'entretien sur l'installation sur {duree_contrat} ans : {cout_total_entretien:,.0f} €".replace(',', ' '))

    # Affichage de la production estimée
    print(f"Production de l'installation photovoltaïque estimée par an : {prod :,.0f} kWh".replace(',',' '))

    print("="*70)
    return temps_payback, benefice_total