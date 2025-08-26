import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def lagrange_interpolation(x_points, y_points):
   
    x = sp.Symbol('x')  # Déclare la variable symbolique pour créer le polynôme

    # Fonction pour calculer la base de Lagrange
    def lagrange_basis(i):
        L_i = 1
        for j in range(len(x_points)):
            if i != j:
                L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return L_i

    # Calcul de l'expression symbolique du polynôme
    polynomial_expr = sum(y_points[i] * lagrange_basis(i) for i in range(len(x_points)))
    
    # Conversion en une fonction Python numérique
    polynomial_func = sp.lambdify(x, polynomial_expr, 'numpy')

    return polynomial_func, polynomial_expr

def plot_polynomial(polynomial_func, start, end):
    """
    Trace le polynôme entre deux valeurs de i (start et end).
    
    Parameters:
    polynomial_func (function): La fonction polynomiale à tracer.
    start (int): La valeur de départ de i (abscisse).
    end (int): La valeur de fin de i (abscisse).
    """
    # Générer les valeurs de i
    i_values = np.arange(start, end, 1)
    P = []

    # Calculer les valeurs du polynôme pour chaque valeur de i
    for i in i_values:
        P.append(polynomial_func(i))

    # Tracer le graphique
    plt.plot(i_values, P, label='Polynôme interpolé')

    # Configuration du graphique
    plt.xlabel('Surface (m²)', fontsize=14)
    plt.ylabel('Ratio eu/m²', fontsize=14)
    plt.title('Graphique du prix de l\'installation en fonction de sa taille', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Demander à l'utilisateur d'entrer les points
n = int(input("Entrez le degré du polynôme (n) : "))
x_points = []
y_points = []

for i in range(n + 1):
    x = float(input(f"Entrez l'abscisse x_{i} : "))
    y = float(input(f"Entrez l'ordonnée y_{i} : "))
    x_points.append(x)
    y_points.append(y)

# Créer le polynôme interpolateur
polynomial_func, polynomial_expr = lagrange_interpolation(x_points, y_points)

# Afficher l'expression du polynôme
print(f"Le polynôme d'interpolation est : {sp.simplify(polynomial_expr)}")

# Afficher le résultat pour différentes valeurs de x
x_eval = float(input("Entrez une valeur de x pour évaluer le polynôme : "))
print(f"P({x_eval}) = {polynomial_func(x_eval)}")

# Tracer le polynôme
plot_polynomial(polynomial_func, start=0, end=239)