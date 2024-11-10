from math import *
import numpy as np

#LA PARTE A) DE ESTA PRIMERA PARTE FUE REALIZADA EN HOJA. LEER INFORME PARA MÁS DETALLES.

#--------------------------- ITEM (B) ---------------------------
#Mediciones EXACTAS
P1 = 106716
P2 = 106031
P = (P1 + P2) / 2
x0 = -P/3500
x1 = P/3500
y0 = y1 = P/2100
L = P/1300


#Por Newton-Raphson, hallamos mu despejando la misma de la ecuación 4), dado que en este caso se cumplen las condiciones
def f(mu):
    return (np.sinh(mu * x1) - np.sinh(mu * x0)) / mu - L

# Derivada de f(mu)
def df(mu):
    return ((mu * (x1 * np.cosh(mu * x1) - x0 * np.cosh(mu * x0))) - 
            (np.sinh(mu * x1) - np.sinh(mu * x0))) / (mu ** 2)

# Valor inicial para mu y tolerancia
mu = 0.1  # Semilla inicial (puede ajustarse este valor)
tolerancia = 0.5e-6

# Newton-Raphson loop
iteration = 0
while True:
    mu_new = mu - f(mu) / df(mu)
    if abs(mu_new - mu) < tolerancia:
        break
    mu = mu_new
    iteration += 1

mu = float(f"{mu:.5g}") #para redondear a 6 decimales significativos

#Cálculo del C2
C2 = y1*mu - (np.cosh(mu*x1)) #despejando de la ecuación (3)

#propagación de errores c2 -> mu tiene error inherente para su cálculo

dc2_mu = y1 - (x1*np.sinh(mu*x1)) #derivada respecto a mu de C2
ec2 = float(f"{(abs(dc2_mu) * abs(tolerancia)):.1g}" ) #cota de C2 ya redondeada
t = math.ceil(-math.log10(2 * ec2)) #decimales significativos

C2 = float(f"{C2:.{t}}")

#para la ecuacion original, mu y C2 como datos de entrada tienen error inherente
def catenaria(x):
    return ((np.cosh(mu*x))/mu) + C2

def df_catenaria_mu(x):
    return ((x*mu*np.sinh(mu*x)) - np.cosh(mu*x))/mu*mu

#y la derivada respecto de C2 es 1 -> se propaga su error como es de entrada
y_x_0 = catenaria(0)
e_y = (abs(df_catenaria_mu(0)) * tolerancia) + 1*ec2
t = math.ceil(-math.log10(2 * ec2))
y_x_0 = float(f"{y_x_0:.{t}}")

print("------------- RESULTADOS BIEN REDONDEADOS ---------")
print("Valor de mu: ", mu)
print("Valor de C2: ", C2)
print("Valor de y(x=0): ", y_x_0)


#--------------------------- ITEM (C) ---------------------------
#A priori, los cálculos serían los mismos. Solo se modifican sus errores

