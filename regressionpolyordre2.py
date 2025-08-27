import numpy as np
import matplotlib.pyplot as plt

def polynomial_regression(x_points, y_points, degree=2):
    
    # Calcul des coefficients du polynôme d'ordre "degree"
    coeffs = np.polyfit(x_points, y_points, degree)

    # Créer la fonction polynomiale à partir des coefficients
    polynomial = np.poly1d(coeffs)

    return polynomial

def plot_polynomial_with_point(polynomial, x_points, y_points, S_instal):
    
    # Générer des valeurs de x pour tracer le polynôme
    x_values = np.linspace(min(x_points), max(x_points), 100)
    y_values = polynomial(x_values)

    # Calculer le prix pour la surface d'installation donnée
    prix_instal = polynomial(S_instal)

    # Tracer les points d'origine
    plt.scatter(x_points, y_points, color='red', label="Sites PV données pour l'estimation")

    # Tracer la courbe du polynôme
    plt.plot(x_values, y_values, label=f"Régression polynomiale d\'ordre 2")

    # Tracer le point vert pour la surface d'installation donnée
    plt.scatter(S_instal, prix_instal, color='green', s=100, marker='o', label=f'Nouvelle installation photovoltaique={S_instal} m²')

    # Configuration du graphique
    plt.xlabel('Surface PV (m²)', fontsize=14)
    plt.ylabel("Prix de l'installation (€)", fontsize=14)
    plt.title("Courbe d'estimation du cout d'une centrale PV", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Affichage du graphique
    plt.show()

