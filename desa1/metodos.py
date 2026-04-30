"""
metodos.py — Biblioteca de métodos numéricos para sistemas de ecuaciones lineales
==================================================================================
Proyecto: Métodos Numéricos para Sistemas de Ecuaciones Lineales (n ≥ 3)
Contenido:
  - Factorización LU (con pivoteo parcial)
  - Método de Jacobi
  - Método de Gauss-Seidel
  - Método SOR (Successive Over-Relaxation)
  - Gradiente Conjugado Precondicionado (PCG)
  - Número de condición y análisis de estabilidad
  - Verificación de convergencia
Nivel: Universitario — Métodos Numéricos
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List




def numero_condicion(A: np.ndarray) -> float:
    """
    Calcula el número de condición de la matriz A usando la norma-2.
    Un número de condición grande indica mal condicionamiento.
    cond(A) = ||A|| * ||A^{-1}||
    """
    return float(np.linalg.cond(A))


def es_diagonal_dominante(A: np.ndarray) -> bool:
    """
    Verifica si la matriz A es estrictamente diagonal dominante por filas.
    Condición suficiente (no necesaria) de convergencia para Jacobi y Gauss-Seidel.
    |a_ii| > sum_{j≠i} |a_ij|  para todo i
    """
    n = A.shape[0]
    for i in range(n):
        suma_fila = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= suma_fila:
            return False
    return True


def radio_espectral(B: np.ndarray) -> float:
    """
    Calcula el radio espectral de la matriz de iteración B.
    rho(B) = max |lambda_i|
    Si rho(B) < 1, el método iterativo converge.
    """
    eigenvalores = np.linalg.eigvals(B)
    return float(np.max(np.abs(eigenvalores)))


def error_relativo(x_aprox: np.ndarray, x_exacto: np.ndarray) -> float:
    """
    Calcula el error relativo entre la solución aproximada y la exacta.
    err = ||x_aprox - x_exacto||_2 / ||x_exacto||_2
    """
    norma_exacta = np.linalg.norm(x_exacto)
    if norma_exacta < 1e-15:
        return float(np.linalg.norm(x_aprox - x_exacto))
    return float(np.linalg.norm(x_aprox - x_exacto) / norma_exacta)


def residuo(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la norma del residuo: ||Ax - b||_2
    Mide cuán bien la solución x satisface el sistema original.
    """
    return float(np.linalg.norm(A @ x - b))




def factorizacion_lu(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Resuelve el sistema Ax = b mediante factorización LU con pivoteo parcial.

    Descomposición: PA = LU
      P: matriz de permutación
      L: triangular inferior con 1s en la diagonal
      U: triangular superior

    Resolución en dos pasos:
      1) Ly = Pb  (sustitución hacia adelante)
      2) Ux = y   (sustitución hacia atrás)

    Parámetros:
        A (ndarray): Matriz de coeficientes n×n
        b (ndarray): Vector de términos independientes

    Retorna:
        x (ndarray): Solución del sistema
        info (dict): Diccionario con L, U, P, residuo y error
    """
    n = len(b)
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    # ── Construir matrices L, U, P ──
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # Pivoteo parcial: buscar el mayor elemento en la columna k
        idx_pivot = np.argmax(np.abs(U[k:, k])) + k

        if abs(U[idx_pivot, k]) < 1e-15:
            raise ValueError(f"Matriz singular o casi singular en columna {k}.")

        if idx_pivot != k:
            # Intercambiar filas en U y P
            U[[k, idx_pivot], :] = U[[idx_pivot, k], :]
            P[[k, idx_pivot], :] = P[[idx_pivot, k], :]
            # Intercambiar parte ya procesada de L
            if k > 0:
                L[[k, idx_pivot], :k] = L[[idx_pivot, k], :k]

        # Eliminación gaussiana
        for i in range(k + 1, n):
            if abs(U[k, k]) < 1e-15:
                continue
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    # ── Sustitución hacia adelante: Ly = Pb ──
    Pb = P @ b
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # ── Sustitución hacia atrás: Ux = y ──
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
            raise ValueError(f"División por cero en U[{i},{i}]. Sistema singular.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    info = {
        "L": L,
        "U": U,
        "P": P,
        "residuo": residuo(A, x, b),
        "iteraciones": 1,   # método directo: equivalente a 1 "paso"
        "convergio": True,
        "errores": [residuo(A, x, b)],
    }
    return x, info




def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 2000,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Resuelve Ax = b mediante el método de Jacobi.

    Esquema de iteración:
      x_i^{(k+1)} = (b_i - sum_{j≠i} a_ij * x_j^{(k)}) / a_ii

    Forma matricial:
      x^{(k+1)} = D^{-1}(b - (L+U)x^{(k)})
      Matriz de iteración: B_J = -D^{-1}(L+U)

    Parámetros:
        A       : Matriz n×n
        b       : Vector n
        tol     : Tolerancia de convergencia (norma del residuo)
        max_iter: Máximo número de iteraciones
        x0      : Aproximación inicial (default: vector de ceros)

    Retorna:
        x (ndarray | None): Solución o None si no converge
        info (dict): historial de errores, iteraciones, convergencia
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    # Verificar que la diagonal no tenga ceros
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(f"Elemento diagonal A[{i},{i}] es cero. No se puede aplicar Jacobi directamente.")

    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    historial_errores = []
    convergio = False

    for k in range(max_iter):
        x_nuevo = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_nuevo[i] = (b[i] - suma) / A[i, i]

        err = float(np.linalg.norm(x_nuevo - x))
        historial_errores.append(err)

        x = x_nuevo.copy()

        if err < tol:
            convergio = True
            break

    info = {
        "iteraciones": len(historial_errores),
        "errores": historial_errores,
        "convergio": convergio,
        "residuo": residuo(A, x, b),
    }
    return x if convergio else x, info




def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 2000,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Resuelve Ax = b mediante el método de Gauss-Seidel.

    Esquema de iteración (usa los valores más recientes):
      x_i^{(k+1)} = (b_i - sum_{j<i} a_ij * x_j^{(k+1)}
                            - sum_{j>i} a_ij * x_j^{(k)}) / a_ii

    Forma matricial:
      x^{(k+1)} = (D+L)^{-1}(b - Ux^{(k)})
      Matriz de iteración: B_GS = -(D+L)^{-1}U

    Ventaja sobre Jacobi: Convergencia típicamente más rápida
    (aprox. el doble de rápida para matrices DD).

    Parámetros:
        A       : Matriz n×n
        b       : Vector n
        tol     : Tolerancia
        max_iter: Máximo de iteraciones
        x0      : Aproximación inicial
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(f"Diagonal nula en A[{i},{i}].")

    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    historial_errores = []
    convergio = False

    for k in range(max_iter):
        x_anterior = x.copy()
        for i in range(n):
            suma_ant = sum(A[i, j] * x[j] for j in range(i))          # ya actualizados
            suma_sig = sum(A[i, j] * x_anterior[j] for j in range(i + 1, n))  # anteriores
            x[i] = (b[i] - suma_ant - suma_sig) / A[i, i]

        err = float(np.linalg.norm(x - x_anterior))
        historial_errores.append(err)

        if err < tol:
            convergio = True
            break

    info = {
        "iteraciones": len(historial_errores),
        "errores": historial_errores,
        "convergio": convergio,
        "residuo": residuo(A, x, b),
    }
    return x, info



def sor(
    A: np.ndarray,
    b: np.ndarray,
    omega: float = 1.25,
    tol: float = 1e-10,
    max_iter: int = 2000,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Resuelve Ax = b mediante el método SOR.

    SOR generaliza Gauss-Seidel introduciendo el parámetro de relajación ω:
      x_i^{(k+1)} = (1-ω)*x_i^{(k)} + ω*(b_i - sum_{j<i} a_ij*x_j^{(k+1)}
                                               - sum_{j>i} a_ij*x_j^{(k)}) / a_ii

    Parámetro ω:
      ω = 1   → Gauss-Seidel exacto
      0 < ω < 1 → Sub-relajación (útil para matrices no-DD)
      1 < ω < 2 → Sobre-relajación (acelera convergencia; elegir ω ≈ 2/(1+sin(π/n)))
      ω ≥ 2 → Diverge siempre (teorema de Kahan)

    Selección óptima de ω (para matrices tridiagonales):
      ω_opt = 2 / (1 + sqrt(1 - rho(B_J)^2))

    Parámetros:
        A     : Matriz n×n
        b     : Vector n
        omega : Parámetro de relajación (1 < ω < 2 para sobre-relajación)
        tol   : Tolerancia
        max_iter : Máximo de iteraciones
        x0    : Aproximación inicial
    """
    if not (0 < omega < 2):
        raise ValueError(f"ω={omega} fuera del rango (0, 2). SOR diverge garantizado.")

    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    historial_errores = []
    convergio = False

    for k in range(max_iter):
        x_anterior = x.copy()
        for i in range(n):
            suma_ant = sum(A[i, j] * x[j] for j in range(i))
            suma_sig = sum(A[i, j] * x_anterior[j] for j in range(i + 1, n))
            x_gs = (b[i] - suma_ant - suma_sig) / A[i, i]   # paso Gauss-Seidel
            x[i] = (1 - omega) * x_anterior[i] + omega * x_gs

        err = float(np.linalg.norm(x - x_anterior))
        historial_errores.append(err)

        if err < tol:
            convergio = True
            break

    info = {
        "iteraciones": len(historial_errores),
        "errores": historial_errores,
        "convergio": convergio,
        "residuo": residuo(A, x, b),
        "omega": omega,
    }
    return x, info


def gradiente_conjugado_precondicionado(
    A: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 2000,
    precondicionador: str = "jacobi",
    x0: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Resuelve Ax = b mediante el Método del Gradiente Conjugado Precondicionado.

    Requiere que A sea simétrica definida positiva (SDP).
    Si no lo es, se trabaja con A^T A x = A^T b (sistema normal).

    Algoritmo PCG:
      Dado precondicionador M ≈ A:
      1) r_0 = b - Ax_0
      2) z_0 = M^{-1} r_0
      3) p_0 = z_0
      4) Para k = 0, 1, 2, ...:
           α_k = (r_k^T z_k) / (p_k^T A p_k)
           x_{k+1} = x_k + α_k p_k
           r_{k+1} = r_k - α_k A p_k
           z_{k+1} = M^{-1} r_{k+1}
           β_k = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
           p_{k+1} = z_{k+1} + β_k p_k

    Precondicionadores disponibles:
      'jacobi': M = diag(A)  — simple, bajo costo
      'ilu'   : M ≈ ILU(0)   — más efectivo para matrices dispersas

    Parámetros:
        A                : Matriz n×n SDP (o se simetriza automáticamente)
        b                : Vector n
        tol              : Tolerancia
        max_iter         : Máximo de iteraciones
        precondicionador : 'jacobi' | 'ilu'
        x0               : Aproximación inicial
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    # Verificar si A es simétrica; si no, usar sistema normal A^T A x = A^T b
    es_simetrica = np.allclose(A, A.T, atol=1e-8)
    if not es_simetrica:
        A_orig = A.copy()
        b_orig = b.copy()
        A = A_orig.T @ A_orig
        b = A_orig.T @ b_orig

    # ── Construir precondicionador ──
    if precondicionador == "jacobi":
        d = np.diag(A)
        d = np.where(np.abs(d) < 1e-15, 1.0, d)
        def M_inv(r):
            return r / d
    elif precondicionador == "ilu":
        # ILU(0) simplificado: factorización incompleta
        M = A.copy()
        for k in range(n):
            for i in range(k + 1, n):
                if abs(M[k, k]) > 1e-15:
                    M[i, k] /= M[k, k]
                for j in range(k + 1, n):
                    M[i, j] -= M[i, k] * M[k, j]
        L_ilu = np.tril(M, -1) + np.eye(n)
        U_ilu = np.triu(M)
        def M_inv(r):
            # Ly = r, Ux = y
            y = np.linalg.solve(L_ilu, r)
            return np.linalg.solve(U_ilu, y)
    else:
        def M_inv(r):
            return r   # Sin precondicionador (CG estándar)

    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    r = b - A @ x
    z = M_inv(r)
    p = z.copy()
    rz = float(r @ z)
    historial_errores = []
    convergio = False

    for k in range(max_iter):
        Ap = A @ p
        pAp = float(p @ Ap)

        if abs(pAp) < 1e-30:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        err = float(np.linalg.norm(r))
        historial_errores.append(err)

        if err < tol:
            convergio = True
            break

        z = M_inv(r)
        rz_nuevo = float(r @ z)
        beta = rz_nuevo / rz
        p = z + beta * p
        rz = rz_nuevo

    info = {
        "iteraciones": len(historial_errores),
        "errores": historial_errores,
        "convergio": convergio,
        "residuo": residuo(A, x, b),
        "precondicionador": precondicionador,
    }
    return x, info




def resolver_todos_los_metodos(
    A: np.ndarray,
    b: np.ndarray,
    nombre_sistema: str = "Sistema",
    tol: float = 1e-10,
    max_iter: int = 2000,
    omega_sor: float = 1.25,
) -> dict:
    """
    Aplica todos los métodos numéricos a un sistema Ax = b y devuelve
    un diccionario con los resultados comparativos.

    Retorna:
        resultados (dict): clave = nombre del método, valor = {x, info}
    """
    x_exacto = np.linalg.solve(A, b)
    resultados = {"x_exacto": x_exacto, "nombre": nombre_sistema}
    cond = numero_condicion(A)
    resultados["condicion"] = cond
    resultados["diagonal_dominante"] = es_diagonal_dominante(A)

    metodos = {
        "LU": lambda: factorizacion_lu(A, b),
        "Jacobi": lambda: jacobi(A, b, tol=tol, max_iter=max_iter),
        "Gauss-Seidel": lambda: gauss_seidel(A, b, tol=tol, max_iter=max_iter),
        f"SOR(ω={omega_sor})": lambda: sor(A, b, omega=omega_sor, tol=tol, max_iter=max_iter),
        "PCG-Jacobi": lambda: gradiente_conjugado_precondicionado(
            A, b, tol=tol, max_iter=max_iter, precondicionador="jacobi"
        ),
    }

    for nombre, fn in metodos.items():
        try:
            x_sol, info = fn()
            info["error_vs_exacto"] = error_relativo(x_sol, x_exacto)
            resultados[nombre] = {"x": x_sol, "info": info}
        except Exception as e:
            resultados[nombre] = {"x": None, "info": {"error_msg": str(e), "convergio": False}}

    return resultados




def graficar_convergencia(resultados: dict, guardar: Optional[str] = None) -> None:
    """
    Genera gráficas de convergencia (iteración vs error) para todos los
    métodos iterativos aplicados a un sistema dado.
    """
    metodos_iterativos = ["Jacobi", "Gauss-Seidel", "PCG-Jacobi"]
    # Incluir SOR con cualquier omega
    for k in resultados:
        if k.startswith("SOR"):
            metodos_iterativos.append(k)

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6", "#F39C12"]
    marcadores = ["o", "s", "^", "D", "v"]

    for idx, metodo in enumerate(metodos_iterativos):
        if metodo not in resultados:
            continue
        info = resultados[metodo]["info"]
        errores = info.get("errores", [])
        if not errores:
            continue
        iters = list(range(1, len(errores) + 1))
        # Escala logarítmica para ver la convergencia exponencial
        ax.semilogy(
            iters,
            errores,
            label=f"{metodo} ({len(errores)} iter)",
            color=colores[idx % len(colores)],
            marker=marcadores[idx % len(marcadores)],
            markevery=max(1, len(iters) // 15),
            linewidth=2,
        )

    ax.set_xlabel("Iteración", fontsize=12)
    ax.set_ylabel("Error ||x^(k+1) - x^(k)||₂ (escala log)", fontsize=12)
    ax.set_title(
        f"Convergencia de Métodos Iterativos — {resultados.get('nombre', '')}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()

    if guardar:
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def graficar_planos_3d(
    A: np.ndarray,
    b: np.ndarray,
    x_sol: np.ndarray,
    titulo: str = "Sistema 3×3 — Intersección de planos",
    guardar: Optional[str] = None,
) -> None:
    """
    Visualiza en 3D los tres planos definidos por Ax = b y su punto de intersección.
    Solo aplicable cuando n = 3.
    """
    if A.shape[0] != 3:
        print("La visualización 3D solo está disponible para sistemas 3×3.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection="3d")

    rango = max(abs(x_sol)) * 2.5 + 3
    u = np.linspace(-rango, rango, 40)
    v = np.linspace(-rango, rango, 40)
    U, V = np.meshgrid(u, v)

    colores_planos = ["#3498DB", "#E74C3C", "#2ECC71"]
    nombres_planos = [
        f"{A[0,0]:.1f}x₁ + {A[0,1]:.1f}x₂ + {A[0,2]:.1f}x₃ = {b[0]:.1f}",
        f"{A[1,0]:.1f}x₁ + {A[1,1]:.1f}x₂ + {A[1,2]:.1f}x₃ = {b[1]:.1f}",
        f"{A[2,0]:.1f}x₁ + {A[2,1]:.1f}x₂ + {A[2,2]:.1f}x₃ = {b[2]:.1f}",
    ]

    for i in range(3):
        a1, a2, a3 = A[i, 0], A[i, 1], A[i, 2]
        bi = b[i]
        try:
            if abs(a3) > 1e-10:
                W = (bi - a1 * U - a2 * V) / a3
                ax3d.plot_surface(
                    U, V, W,
                    alpha=0.35,
                    color=colores_planos[i],
                    label=nombres_planos[i],
                )
            elif abs(a2) > 1e-10:
                W = (bi - a1 * U - a3 * V) / a2
                ax3d.plot_surface(U, W, V, alpha=0.35, color=colores_planos[i])
        except Exception:
            pass

    # Punto de intersección
    ax3d.scatter(
        [x_sol[0]], [x_sol[1]], [x_sol[2]],
        color="gold", s=200, zorder=10, label=f"Solución: ({x_sol[0]:.3f}, {x_sol[1]:.3f}, {x_sol[2]:.3f})"
    )

    ax3d.set_xlabel("x₁", fontsize=11)
    ax3d.set_ylabel("x₂", fontsize=11)
    ax3d.set_zlabel("x₃", fontsize=11)
    ax3d.set_title(titulo, fontsize=12, fontweight="bold", pad=15)
    ax3d.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def imprimir_tabla_comparativa(todos_resultados: List[dict]) -> None:
    """
    Imprime en consola una tabla comparativa de todos los métodos
    para los tres sistemas analizados.
    """
    print("\n" + "═" * 90)
    print("TABLA COMPARATIVA DE MÉTODOS NUMÉRICOS".center(90))
    print("═" * 90)

    nombres_sistemas = [r.get("nombre", f"S{i+1}") for i, r in enumerate(todos_resultados)]
    print(f"{'Método':<28} | {'Iter (Ideal)':>12} | {'Iter (Stress)':>13} | "
          f"{'Iter (MalCond)':>14} | {'Converge':>8}")
    print("-" * 90)

    metodos_lista = ["LU", "Jacobi", "Gauss-Seidel", "PCG-Jacobi"]
    for r in todos_resultados:
        for k in r:
            if k.startswith("SOR"):
                if k not in metodos_lista:
                    metodos_lista.append(k)
                break

    for metodo in metodos_lista:
        fila = f"{metodo:<28}"
        for r in todos_resultados:
            if metodo in r:
                info = r[metodo]["info"]
                iters = info.get("iteraciones", "—")
                fila += f" | {str(iters):>13}"
            else:
                fila += f" | {'N/A':>13}"

        # Convergencia (basada en el primer sistema)
        if metodo in todos_resultados[0]:
            conv = "Sí ✓" if todos_resultados[0][metodo]["info"].get("convergio") else "No ✗"
        else:
            conv = "—"
        fila += f" | {conv:>8}"
        print(fila)

    print("═" * 90)

    # Números de condición
    print("\nNúmeros de condición:")
    for r in todos_resultados:
        cond = r.get("condicion", float("nan"))
        dd = "Sí" if r.get("diagonal_dominante") else "No"
        print(f"  {r.get('nombre', '?'):<35} κ(A) = {cond:.2e}   Diag. Dom. = {dd}")
    print()
