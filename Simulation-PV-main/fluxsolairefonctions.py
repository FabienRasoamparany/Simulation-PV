import math
import numpy as np

#-----------------------------Calcul H0----------------------------------#
def H0(jour,mois, latitude): #energie recu sur 1m² horizontal pdt 1 journée

    if mois<3:
        jour_N=jour+31*(mois-1)
    else:
        jour_N=jour+31*(mois-1)-int(0.4*mois+2.3)
    
    declinaison_solaire = math.radians(23.45 * math.sin(math.radians((360/365) * (jour_N - 81))))#en radians
    angle_horaire0 = math.acos(-math.tan(math.radians(latitude)) * math.tan(declinaison_solaire))#en radians
    angle_zenithal = math.cos(math.radians(latitude))*math.cos(declinaison_solaire)*math.sin(angle_horaire0)+math.sin(declinaison_solaire)*math.sin(math.radians(latitude))*angle_horaire0

    constante_solaire = 1367 * (1 + 0.033 * math.cos(math.radians(360 * jour_N / 365)))

    irradiance_solaire = (24*constante_solaire/math.pi) * angle_zenithal

    return irradiance_solaire

#-----------------------------Calcul Kt----------------------------------#   
def f(x, coeffs):
    return coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]

def df(x, coeffs):
    return 3 * coeffs[0] * x**2 + 2 * coeffs[1] * x + coeffs[2]

def newton_raphson(coefficients, x0, tolerance=1e-6, max_iterations=100):
    x = x0
    iterations = 0
    while True:
        x1 = x - f(x, coefficients) / df(x, coefficients)
        if abs(x1 - x) < tolerance or iterations >= max_iterations:
            break
        x = x1
        iterations += 1
    if iterations >= max_iterations:
        print("La méthode n'a pas convergé après", max_iterations, "itérations.")
    
    return x1

#-----------------------------Calcul H0 moyen----------------------------------# 
def H0_moyen():
    jours_moyenne=[17, 16, 16, 15, 15, 11, 17, 16, 15, 15, 14, 10]
    mois=np.arange(1,13)
    H0_moyen=[None]*12
    for o in range(0,12):
        H0_moyen[o]=H0(jours_moyenne[o],mois[o],43.6)
    return H0_moyen

#-----------------------------Calcul angle horaire----------------------------------# 
def angle_horaire(h):
    w=15*(h-12)
    w=math.radians(w)
    return w 

#-----------------------------Calcul rd----------------------------------# 
def rd_dh(h,jour,mois,latitude): 
    jour_N = Jour_N(jour,mois)
    declinaison = declinaison_solaire(jour_N)
    anglehoraire0 = math.degrees(angle_horaire0(jour,mois,latitude))
    w=15*(h-12)
    w=math.radians(w)
    rd=(math.pi/24)*((math.cos(w)-math.cos(math.radians(anglehoraire0)))/(math.sin(math.radians(anglehoraire0))-(anglehoraire0*math.pi/180)*math.cos(math.radians(anglehoraire0))))
    return rd

def rd_gh(h,jour,mois,latitude):
    jour_N = Jour_N(jour,mois)
    declinaison = declinaison_solaire(jour_N)
    anglehoraire0 = angle_horaire0(jour,mois,latitude)
    w=angle_horaire(h)
    a=0.409+0.5016*math.sin(anglehoraire0-math.pi/3)
    b=0.6609-0.4767*math.sin(anglehoraire0-math.pi/3)
    rd_gh=(a+b*math.cos(math.radians(w)))*rd_dh(h,jour,mois,latitude)
    return rd_gh

#-----------------------------Calcul altitude solaire h----------------------------------# 
def altitude_solaire(latitude, declinaison, angle_horaire):
    return math.degrees(math.asin(math.sin(math.radians(latitude)) * math.sin(declinaison) + math.cos(math.radians(latitude)) * math.cos(declinaison) * math.cos(angle_horaire)))

#-----------------------------Calcul l'azimut solaire----------------------------------#
def azimut_solaire(latitude, declinaison, angle_horaire):
    alt_solaire = altitude_solaire(latitude, declinaison, angle_horaire)
    if alt_solaire > 0:
        # Convertir latitude en radians pour les calculs trigonométriques
        latitude_rad = math.radians(latitude)
        
        # Calculer sin et cos de l'azimut en utilisant des radians
        sin_azimut = (math.cos(declinaison) * math.sin(angle_horaire)) / math.cos(math.radians(alt_solaire))
        cos_azimut = (math.sin(declinaison) - math.sin(latitude_rad) * math.sin(math.radians(alt_solaire))) / (math.cos(latitude_rad) * math.cos(math.radians(alt_solaire)))
        
        # Calculer l'azimut en degrés
        azimut = math.degrees(math.atan2(sin_azimut, cos_azimut))
        if azimut < 0:
            azimut += 360
        return azimut
    else:
        return 0
#----------------------------Calcul déclinaison------------------------------#
def declinaison_solaire(jour_N):
    return math.radians(23.45 * math.sin(math.radians((360 / 365) * (jour_N - 81))))
    
#-----------------------------Calcul l'angle indicent solaire pour surf inclinée----------------------------------#
def angle_incident(azimut_surface,inclinaison_surface,azimut,elevation):
    angle_incident = math.acos(math.cos(elevation)*math.cos(azimut-azimut_surface)*math.sin(inclinaison_surface)*math.sin(elevation))

#-----------------------------Calcul A B et C----------------------------------#
def coeff_X(inclinaison, jour, mois, latitude, azimut_surface, coeff):
    jour_N = Jour_N(jour, mois)
    declinaison = declinaison_solaire(jour_N)

    # Convertir les angles en radians pour les calculs
    i_rad = math.radians(inclinaison)
    latitude_rad = math.radians(latitude)
    azimut_surface_rad = math.radians(azimut_surface)
    
    A = math.cos(i_rad) + math.tan(latitude_rad) * math.cos(azimut_surface_rad) * math.sin(i_rad)
    B = math.cos(angle_horaire0(jour, mois, latitude)) * math.cos(i_rad) + math.tan(declinaison) * math.sin(i_rad) * math.cos(azimut_surface_rad)
    C = (math.cos(i_rad) * math.sin(azimut_surface_rad)) / math.cos(latitude_rad)
    
    if coeff == 'A':
        return A
    elif coeff == 'B':
        return B
    elif coeff == 'C':
        return C
    else:
        raise ValueError("Invalid coefficient requested. Use 'A', 'B', or 'C'.")
#----------------------------Calcul angle horaire 0 ------------------------------#
def angle_horaire0(jour, mois, latitude):
    jour_N = Jour_N(jour, mois)
    declinaison = declinaison_solaire(jour_N)
    
    # Calcul de l'angle horaire 0 en radians
    latitude_rad = math.radians(latitude)
    angle_horaire0_rad = math.acos(-math.tan(latitude_rad) * math.tan(declinaison))
    return angle_horaire0_rad
#----------------------------Calcul Jour N ------------------------------#
def Jour_N(jour,mois):
    if mois<3:
        jour_N=jour+31*(mois-1)
    else:
        jour_N=jour+31*(mois-1)-int(0.4*mois+2.3)
    return jour_N

#----------------------------Fonction G(wss,wsr) ------------------------------#
def G(w1, w2, jour, mois, latitude, gmoyen, dmoyen, A, B, C):
    w0 = angle_horaire0(jour, mois, latitude)
    
    # Calculer d, a, b, aprime en radians
    d = math.sin(w0) - w0 * math.cos(w0) 
    a = 0.409 + 0.5016 * math.sin(w0 - math.radians(60))
    b = 0.6609 - 0.4767 * math.sin(w0 - math.radians(60))
    aprime = a - dmoyen / gmoyen
    
    # Calculer g en utilisant des valeurs angulaires en radians
    g = (1 / (2 * d)) * (
        (b * A / 2 - aprime * B) * (w1 - w2) * math.pi / 180 +
        (aprime * B - b * B) * (math.sin(math.radians(w1)) - math.sin(math.radians(w2))) -
        aprime * C * (math.cos(math.radians(w1)) - math.cos(math.radians(w2))) +
        (b * A / 2) * (math.sin(math.radians(w1)) * math.cos(math.radians(w1)) - math.sin(math.radians(w2)) * math.cos(math.radians(w2)) +
        (b * A / 2) * (math.sin(math.radians(w1))**2 - math.sin(math.radians(w2))**2))
    )
    return g


#----------------------------Fonction G surface inclinée  ------------------------------#
def irradiation_inclined_surface(Gh, Dh, albedo, inclinaison, azimut, latitude, jour, mois, coeff):
    """
    Calcule l'irradiation sur une surface inclinée.

    Parameters:
    Gh (float): Irradiation sur la surface horizontale (Wh/j/m²).
    Dh (float): Composante diffuse de l'irradiation sur la surface horizontale (Wh/j/m²).
    albedo (float): Albédo de la surface (réflectivité).
    inclinaison (float): Angle d'inclinaison de la surface (en degrés).
    azimut (float): Angle d'orientation azimutale de la surface (en degrés).
    latitude (float): Latitude du lieu (en degrés).
    jour (int): Jour du mois.
    mois (int): Mois de l'année.
    coeff (str): Coefficient à retourner ('G', 'S', 'D', 'R').

    Returns:
    float: Irradiation sur la surface inclinée (Wh/j/m²).
    """ 
    irrad_journa = 0
    # Conversion des angles en radians
    inclinaison_rad = math.radians(inclinaison)
    azimut_rad = math.radians(azimut)
    latitude_rad = math.radians(latitude)
    heure = range(6,18)
    for h in heure:
            
        # Angle horaire pour midi solaire
        w = (h - 12) * (math.pi / 12)
        
        # Calcul de la déclinaison solaire
        jour_N = Jour_N(jour, mois)
        declinaison = declinaison_solaire(jour_N)

        # Calcul du cosinus de l'angle d'incidence
        cos_theta = (
            math.sin(declinaison) * math.sin(latitude_rad) * math.cos(inclinaison_rad) - 
        math.sin(declinaison) * math.cos(latitude_rad) * math.sin(inclinaison_rad) * math.cos(azimut_rad) + 
        math.cos(declinaison) * math.cos(latitude_rad) * math.cos(inclinaison_rad) * math.cos(w) +
        math.cos(declinaison) * math.sin(latitude_rad) * math.sin(inclinaison_rad) * math.cos(azimut_rad) * math.cos(w) +
        math.cos(declinaison) * math.sin(inclinaison_rad) * math.sin(azimut_rad) * math.sin(w)
        )

        # Calcul de l'irradiation directe sur la surface inclinée
        G_direct = max(0, Gh * cos_theta)

        # Calcul de l'irradiation diffuse sur la surface inclinée
        G_diffus = Dh * (1 + math.cos(inclinaison_rad)) / 2

        # Calcul de l'irradiation réfléchie sur la surface inclinée
        G_reflected = albedo * Gh * (1 - math.cos(inclinaison_rad)) / 2

        # Irradiation totale sur la surface inclinée
        G_incline = G_direct + G_diffus + G_reflected
        irrad_journa = irrad_journa + G_incline
        
    if coeff == 'G':
        return G_incline
    elif coeff == 'S':
        return G_direct
    elif coeff == 'D':
        return G_diffus
    elif coeff == 'R':
        return G_reflected
    else:
        raise ValueError("Invalid coefficient requested. Use 'G' for global, 'S' for straight, 'D' for diffuse, 'R' for reflected.")
    
def D(h, jour, mois, Dh, latitude):
    rd = rd_dh(h, jour, mois, latitude)
    Dh_horaire = rd * Dh 
    return Dh_horaire

def calcul_radiation_journalier_Dh(jour_ref, mois, Dh, latitude):
    heures = range(24)
    radiation_horaire = []
    for h in heures:
        rd = rd_dh(h, jour_ref, mois, latitude)
        if rd > 0:  # Ne calculez le rayonnement que lorsque rd est positif
            radiation_horaire.append(rd * Dh )
        else:
            radiation_horaire.append(0)  # Pas de rayonnement pendant la nuit
    return radiation_horaire

def calcul_radiation_journalier_Gh(jour_ref, mois, Gh, latitude):
    heures = range(24)
    radiation_horaire = []
    for h in heures:
        rg = rd_gh(h, jour_ref, mois,latitude)
        if rg > 0:
            radiation_horaire.append(rg * Gh )
        else:
            radiation_horaire.append(0)
    return radiation_horaire

def sinus_elevation(h):
    return math.sin(math.radians(h))

def cosinus_teta(h,azimut,azimut_surface,inclinaison):
    return math.cos(math.radians(h))*math.cos(math.radians(azimut-azimut_surface))*math.sin(math.radians(inclinaison))+math.cos(math.radians(inclinaison))*math.sin(math.radians(h))
