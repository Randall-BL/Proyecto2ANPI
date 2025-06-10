import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Parámetros del problema
T_s = 25.0        # Temperatura ambiente (°C)
T0 = 85.0         # Temperatura inicial del CPU (°C)
K = 0.15           # Constante de enfriamiento (1/min)
t_final = 30      # Tiempo total a simular (min)
h = 0.95             # Paso de tiempo (min)

# Función derivada de T según la EDO
def dT_dt(T, T_s, K):
    """
    EDO de la Ley de Enfriamiento de Newton.

    Parámetros:
    T : Temperatura actual del objeto.
    T_s : Temperatura ambiente.
    K : Constante de enfriamiento.

    Retorna:
        La ley de enfriamiento de Newton
    """
    return -K * (T - T_s)

# Método de Euler para resolver la EDO
def euler_method(T0, T_s, K, h, t_final):
    """
    Resuelve la EDO utilizando el Método de Euler.

    Parámetros:
    T0: Temperatura inicial.
    T_s: Temperatura ambiente.
    K: Constante de enfriamiento.
    h: Paso de tiempo.
    t_final: Tiempo total de simulación.

    Retorna:
    tuple: Un par ordenado (t_points, T_points) para cada intervalo de tiempos
    """
    t_points = np.arange(0, t_final + h, h)  # Puntos de tiempo
    T_points = np.zeros_like(t_points)       # Inicialización de temperaturas
    T_points[0] = T0                         # Temperatura inicial
    for i in range(1, len(t_points)):
        # Método de Euler: T[i] = T[i-1] + h * dT/dt
        T_points[i] = T_points[i-1] + h * dT_dt(T_points[i-1], T_s, K)
    return t_points, T_points

def hermite_interpolation_numeric(x, y, dy, x_eval):
    """
    Realiza la interpolación numérica de Hermite para aproximar valores intermedios.

    Parámetros:
    x: Puntos de tiempo conocidos.
    y: Valores de temperatura en los puntos de tiempo.
    dy: Derivadas de la temperatura en los puntos de tiempo.
    x_eval: Puntos de tiempo en los que se evaluará la interpolación.

    Retorna:
        Valores interpolados en los puntos de tiempo.
    """
    n = len(x)
    y_eval = np.zeros_like(x_eval)
    
    def h00(t): return (1 + 2*t) * (1 - t)**2
    def h10(t): return t * (1 - t)**2
    def h01(t): return t**2 * (3 - 2*t)
    def h11(t): return t**2 * (t - 1)
    
    for i, xv in enumerate(x_eval):
        if xv <= x[0]:
            j = 0
        elif xv >= x[-1]:
            j = n - 2
        else:
            j = np.searchsorted(x, xv) - 1
        
        # Intervalo entre los puntos de tiempo
        h_interval = x[j+1] - x[j]
        t = (xv - x[j]) / h_interval
        
        # Interpolación de Hermite
        y_eval[i] = (h00(t) * y[j] + 
                     h10(t) * h_interval * dy[j] + 
                     h01(t) * y[j+1] + 
                     h11(t) * h_interval * dy[j+1])
    return y_eval

def hermite_interpolation_symbolic(x, y, dy):
    """
    Realiza la interpolación simbólica de Hermite con SymPy.

    Parámetros:
    x: Puntos de tiempo conocidos.
    y: Valores de temperatura en los puntos de tiempo.
    dy: Derivadas de la temperatura en los puntos de tiempo.

    Retorna:
        Lista de polinomios simbólicos de Hermite.
    """
    x_sym = sp.symbols('x')
    polys = []
    n = len(x) - 1
    
    for j in range(n):
        h_interval = x[j+1] - x[j]
        t = (x_sym - x[j]) / h_interval
        
        # Definición de las funciones base de Hermite
        h00 = (1 + 2*t) * (1 - t)**2
        h10 = t * (1 - t)**2
        h01 = t**2 * (3 - 2*t)
        h11 = t**2 * (t - 1)
        
        # Polinomio de Hermite para el intervalo
        P = h00 * y[j] + h10 * h_interval * dy[j] + h01 * y[j+1] + h11 * h_interval * dy[j+1]
        P_simplified = sp.simplify(P)
        polys.append((x[j], x[j+1], P_simplified))
    return polys

# Ejecutar el Método de Euler para obtener los pares de tiempo y temperatura
t_points, T_points = euler_method(T0, T_s, K, h, t_final)

# *** PRINT DE LOS PARES ORDENADOS DEL MÉTODO DE EULER ***
print("Pares ordenados obtenidos por el método de Euler:")
print("---------------------------------------------")
print("{:<10} {:<15}".format("Tiempo (min)", "Temperatura (°C)"))
print("---------------------------------------------")
for i in range(len(t_points)):
    print("{:<10.2f} {:<15.4f}".format(t_points[i], T_points[i]))
print("---------------------------------------------")
# ********************************************************
# Calcular las derivadas de la temperatura en los puntos
dT_points = dT_dt(T_points, T_s, K)

# Evaluar la interpolación de Hermite numérica para graficar
t_fine = np.linspace(0, t_final, 300)  # Puntos finos de tiempo para evaluación
T_hermite_num = hermite_interpolation_numeric(t_points, T_points, dT_points, t_fine)

# Solución analítica para la Ley de Enfriamiento de Newton
def T_analytical(t, T_s, T0, K):
    """
    Calcula la solución exacta de la EDO usando la respuesta analítica .

    Parámetros:
    t: Puntos de tiempo.
    T_s: Temperatura ambiente.
    T0: Temperatura inicial.
    K: Constante de enfriamiento.

    Retorna:
        La función evaluada.
    """
    return T_s + (T0 - T_s) * np.exp(-K * t)

# Calcular los valores de la temperatura usando la solución analítica
T_analytical_vals = T_analytical(t_fine, T_s, T0, K)

# Graficar los resultados
plt.plot(t_points, T_points, 'o', label='Pares ordenados (Euler)')  # Método de Euler
plt.plot(t_fine, T_hermite_num, '-', label='Interpolación Hermite (numérica)')  # Interpolación numérica
plt.plot(t_fine, T_analytical_vals, '--', label='Solución analítica', color='r')  # Solución exacta
plt.xlabel('Tiempo (min)')
plt.ylabel('Temperatura (°C)')
plt.title('Ley de Enfriamiento de Newton: Comparación de soluciones')
plt.legend()
plt.grid(True)
plt.show()

# Obtener e imprimir los polinomios simbólicos de Hermite
polys = hermite_interpolation_symbolic(t_points, T_points, dT_points)

print("\nPolinomios de Hermite por intervalos:")
x_sym = sp.symbols('x')
for (a, b, P) in polys:
    print(f"Intervalo [{a}, {b}]:")
    sp.pprint(P)
    print()
