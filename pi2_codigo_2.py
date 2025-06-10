####################################################

# Proyecto 2  
# Análisis Numérico Para La Ingeniería GR 1
# I - Semestre 2025
# Grupo de trabajo # 2
# Integrantes:
# Emanuel Chavarría Hernández
# Fernando Fuchs Mora
# Javier Tenorio Cervantes
# Randall Bolaños López

####################################################

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Parámetros del problema
t0 = 0              # Tiempo inicial (min)
tf = 30             # Tiempo final (min)
h = 0.95            # Paso (min)
T_a = 25            # Temperatura ambiente (°C)
T0 = 85             # Temperatura inicial (°C)
k = 0.15            # Constante de enfriamiento (min⁻¹)

# 1) EDO: dT/dt = -k (T - T_a)
def f(t, T):
    return -k * (T - T_a)

# 2) Implementacion de Runge-Kutta 4 para obtener pares ordenados.
def runge_kutta(f, t0, T0, tf, h):
    """
    Resuelve una EDO de la forma dT/dt = f(t, T) usando el método de Runge-Kutta de cuarto orden (RK4).

    Entradas:
    -----------
    f   : función que devuelve la derivada dT/dt en el punto (t, T).
    t0  : Tiempo inicial.
    T0  : Valor inicial de la variable dependiente T en t = t0.
    tf  : Tiempo final hasta donde se integra.
    h   : Paso de integración (incremento de tiempo).

    Salidas:
    --------
    t_vals : Array con los instantes de tiempo donde se calculó la solución,
             empezando en t0 y avanzando en pasos de tamaño h hasta tf.
    T_vals : Array con los valores aproximados de T en cada uno de los
             tiempos en t_vals, obtenido por RK4.
    """
    t_vals, T_vals = [t0], [T0]
    t, T = t0, T0

    while t < tf:
        # Cálculo de los cuatro incrementos de RK4
        k1 = f(t,               T)
        k2 = f(t + h/2,         T + h*k1/2)
        k3 = f(t + h/2,         T + h*k2/2)
        k4 = f(t + h,           T + h*k3)

        # Actualización de T y t
        T = T + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_vals.append(t)
        T_vals.append(T)

    return np.array(t_vals), np.array(T_vals)


# 3) Hermite por diferencias divididas 
def hermite_divided_differences(x, y, dy):
    """
    Construye la tabla de diferencias divididas para la interpolación de Hermite
    y extrae los coeficientes del polinomio en forma de Newton Hermite.

      1. Duplicación de nodos: z_{2i} = z_{2i+1} = x[i].
      2. Inicialización de Q:
         - Q[2i,   0] = y[i]
         - Q[2i+1, 0] = y[i]
         - Q[2i+1, 1] = dy[i]   (derivada en el nodo duplicado)
         - Q[2i,   1] = (y[i] - y[i-1])/(x[i] - x[i-1]) para i>0
      3. Relleno de la tabla Q para órdenes superiores:
         Q[i,j] = (Q[i,j-1] - Q[i-1,j-1]) / (z[i] - z[i-j])
      4. Los coeficientes del polinomio de Hermite son los elementos de la diagonal Q[i,i].

    Entradas:
    -----------
    x  : Abscisas originales de los nodos.
    y  : Valores de la función f en cada x[i].
    dy : Valores de la derivada f' en cada x[i].

    Salidas:
    --------
    z    : Array con los nodos duplicados: [x0, x0, x1, x1, ..., x_{n-1}, x_{n-1}].
    coef : Coeficientes del polinomio de Hermite en la forma de Newton,
           extraídos de la diagonal de Q.
    """
    n = len(x)
    m = 2 * n
    z = np.zeros(m)
    Q = np.zeros((m, m))

    # 1) nodos duplicados e inicialización de Q[:,0] y Q[:,1]
    for i in range(n):
        z[2*i]   = x[i]
        z[2*i+1] = x[i]
        Q[2*i,   0] = y[i]
        Q[2*i+1, 0] = y[i]
        Q[2*i+1, 1] = dy[i]  # f'[x_i]
        if i > 0:
            # f[x_i, x_{i-1}] = (f(x_i) - f(x_{i-1})) / (x_i - x_{i-1})
            Q[2*i, 1] = (Q[2*i,0] - Q[2*i-1,0]) / (z[2*i] - z[2*i-1])

    # 2) diferencias divididas de orden superior
    for j in range(2, m):
        for i in range(j, m):
            Q[i,j] = (Q[i,j-1] - Q[i-1,j-1]) / (z[i] - z[i-j])

    # 3) coeficientes: diagonal de Q
    coef = np.array([Q[i,i] for i in range(m)])
    return z, coef


def hermite_interpolation(x, y, dy, x_eval):
    """
    Evalúa el polinomio de interpolación de Hermite en los puntos x_eval,
    usando la forma de Newton Hermite obtenida por diferencias divididas.

      - Se duplica cada nodo x[i] para construir el array z.
      - Se calculan los coeficientes coef[k] = f[z0,…,zk] de la diagonal de Q.
      - El polinomio se evalúa como
          H(x) = coef[0]
               + coef[1]*(x - z[0])
               + coef[2]*(x - z[0])*(x - z[1])
               + … 
      - Aquí z y coef se obtienen con hermite_divided_differences().

    Entradas:
    -----------
    x      : Abscisas originales de los nodos.
    y      : Valores de la función f en cada x[i].
    dy     : Valores de la derivada f' en cada x[i].
    x_eval : Puntos donde se desea evaluar el polinomio de Hermite.

    Salidas:
    --------
    y_eval : Valores del polinomio de Hermite en cada punto de x_eval.
    """
    # Obtener nodos duplicados z y coeficientes coef[k]
    z, coef = hermite_divided_differences(x, y, dy)

    # Preparar vector de salida
    y_eval = np.zeros_like(x_eval)

    # Evaluar polinomio en cada x_eval[k]
    for k, xv in enumerate(x_eval):
        # Termino inicial (grado 0)
        val = coef[0]
        # Producto acumulado para (x - z[0])*(x - z[1])*... hasta el grado actual
        prod = 1.0

        # Sumar cada término de Newton–Hermite
        for i in range(1, len(coef)):
            prod *= (xv - z[i-1])     # extiende el producto hasta z[i-1]
            val  += coef[i] * prod   # coef[i] * producto acumulado

        y_eval[k] = val

    return y_eval


# 4) Solución analítica exacta
def T_exact(t):
    return T_a + (T0 - T_a) * np.exp(-k * t)

# 5) Imprimir polinomios de Hermite H_n(x) para n = 0..5
x = sp.symbols('x')
print("Polinomios de Hermite obtenidos H_n(x):")
for n in range(8):
    r = sp.symbols('r', integer=True, nonnegative=True)
    summ = sp.summation(
        (-1)**r * sp.factorial(n)/(sp.factorial(r)*sp.factorial(n-2*r)) * (2*x)**(n-2*r),
        (r, 0, sp.floor(n/2))
    )
    Hn = sp.expand(summ)
    print(f"H_{n}(x) =", Hn)

# 6) Calcular nodos (pares ordenados) con RK4
t_nodes, T_nodes = runge_kutta(f, t0, T0, tf, h)
dTdt_nodes = f(t_nodes, T_nodes)

# 7) Elegir nodos representativos para Hermite (intervalos de 5)

"""
Selección de nodos para construir el polinomio de Hermite.

- Se toma una muestra de los puntos generados por el método de Runge-Kutta de orden 4
  con un paso regular (cada 5 nodos en este caso) para usarlos como nodos de interpolación.

- Lo ideal es reducir la cantidad de nodos usados en la interpolación para:
    * Mejorar la eficiencia computacional.
    * Evitar oscilaciones innecesarias que pueden aparecer si se usan muchos nodos (fenómeno de Runge).
    * Observar gráficamente cómo cambia la aproximación del polinomio de Hermite dependiendo del número de nodos.

- Variables:
    t_h   → nodos en el eje del tiempo seleccionados.
    T_h   → temperaturas correspondientes en los nodos.
    dT_h  → derivadas de la temperatura en esos nodos (obtenidas de la EDO).

- Cambiar el paso (por ejemplo, de 5 a 2) permite comparar visualmente
  cómo afecta la densidad de nodos a la precisión de la interpolación de Hermite.
"""
indices = np.arange(0, len(t_nodes), 5)
t_h = t_nodes[indices]
T_h = T_nodes[indices]
dT_h = dTdt_nodes[indices]

# Imprimir en consola los pares ordenados (t_h, T_h)
print("Pares ordenados utilizados para la Interpolación Hermite (t_i, T_i):")
for ti, Ti in zip(t_h, T_h):
    print(f"({ti:.4f}, {Ti:.4f})")

# LLamar a las funciones de la Solucion Exacta y la Interpolacion de Hermite
t_dense       = np.linspace(t0, tf, 300)
T_dense_exact = T_exact(t_dense)
T_dense_herm  = hermite_interpolation(t_h, T_h, dT_h, t_dense)


# 8) Graficar las Soluciones de la EDO

plt.figure(figsize=(8,5))
# Solución exacta continua
plt.plot(t_dense, T_dense_exact, label='Exacta', linewidth=2)
# Puntos y línea de RK4
plt.scatter(t_nodes, T_nodes, c='blue', marker='o', label='RK4 h=0.95')
plt.plot(t_nodes, T_nodes, c='blue', linewidth=1, alpha=0.5)
# Interpolación de Hermite
plt.plot(t_dense, T_dense_herm, '--', label='Interpolación Hermite', linewidth=2)
plt.scatter(t_h, T_h, c='black', marker='o', label='Puntos Hermite')
# Etiquetas y leyenda
plt.xlabel('Tiempo (min)')
plt.ylabel('Temperatura (°C)')
plt.title('Gráfica 1.Comparación: Solución Exacta vs RK4 vs Hermite')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

