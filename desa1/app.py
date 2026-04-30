

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Importar biblioteca de métodos numéricos
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from metodos import (
    factorizacion_lu, jacobi, gauss_seidel, sor,
    gradiente_conjugado_precondicionado,
    numero_condicion, es_diagonal_dominante, error_relativo, residuo
)


st.set_page_config(
    page_title="Métodos Numéricos — Sistemas Lineales",
    
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .titulo-principal {
        font-size: 2rem; font-weight: 700;
        color: #1a237e; text-align: center;
        padding: 0.5rem 0;
    }
    .subtitulo {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .metrica-card {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #3f51b5;
    }
    .advertencia {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.8rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .exito {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.8rem;
        border-radius: 4px;
    }
    .error-box {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.8rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="titulo-principal"> Métodos Numéricos — Sistemas de Ecuaciones Lineales</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Factorización LU • Jacobi • Gauss-Seidel • SOR • Gradiente Conjugado Precondicionado</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("Configuración")

    # Tamaño del sistema
    n = st.slider("Tamaño del sistema (n × n)", min_value=3, max_value=6, value=3, step=1)

    st.subheader("Sistemas predefinidos")
    sistema_predef = st.selectbox(
        "Cargar sistema:",
        ["Manual", "Caso Ideal (n=3)", "Caso Bajo Estrés (n=3)", "Caso Mal Condicionado (n=3)", "Sistema 4×4"]
    )

    st.divider()
    st.subheader("🔧 Parámetros de los métodos")
    tol = st.number_input("Tolerancia (ε)", value=1e-10, format="%.2e", min_value=1e-16, max_value=1e-3)
    max_iter = st.number_input("Máximo de iteraciones", value=1000, min_value=10, max_value=10000, step=100)
    omega = st.slider("Parámetro SOR (ω)", min_value=0.1, max_value=1.99, value=1.25, step=0.05)

    st.divider()
    st.subheader("Método a resolver")
    metodo_sel = st.selectbox(
        "Seleccionar método:",
        ["Todos", "Factorización LU", "Jacobi", "Gauss-Seidel", f"SOR (ω={omega:.2f})", "PCG-Jacobi"]
    )

    precond_pcg = st.selectbox("Precondicionador PCG:", ["jacobi", "ilu"])

sistemas_predef = {
    "Caso Ideal (n=3)": {
        "A": np.array([[10.0, -1.0, 2.0],
                       [-1.0,  11.0, -1.0],
                       [2.0, -1.0,  10.0]]),
        "b": np.array([6.0, 25.0, -11.0]),
        "desc": "Sistema bien condicionado — Distribución de temperatura en una placa"
    },
    "Caso Bajo Estrés (n=3)": {
        "A": np.array([[500.0, -200.0, 100.0],
                       [-200.0, 800.0, -150.0],
                       [100.0, -150.0, 600.0]]),
        "b": np.array([1500.0, -2000.0, 3000.0]),
        "desc": "Coeficientes grandes — Análisis estructural con cargas elevadas"
    },
    "Caso Mal Condicionado (n=3)": {
        "A": np.array([[1.0, 1.0, 1.0],
                       [1.0, 1.001, 1.0],
                       [1.0, 1.0, 1.001]]),
        "b": np.array([3.0, 3.001, 3.001]),
        "desc": "Ecuaciones casi dependientes — Número de condición muy alto"
    },
    "Sistema 4×4": {
        "A": np.array([[8.0, -2.0, 1.0, 0.0],
                       [-2.0, 9.0, -2.0, 1.0],
                       [1.0, -2.0, 8.0, -2.0],
                       [0.0, 1.0, -2.0, 7.0]]),
        "b": np.array([10.0, -5.0, 20.0, 15.0]),
        "desc": "Sistema 4×4 — Diagonal dominante"
    },
}

st.subheader("Entrada del Sistema Ax = b")

col_a, col_b = st.columns([3, 1])

with col_a:
    st.markdown("**Matriz A**")

    # Inicializar valores según sistema predefinido
    if sistema_predef != "Manual" and sistema_predef in sistemas_predef:
        A_default = sistemas_predef[sistema_predef]["A"]
        b_default = sistemas_predef[sistema_predef]["b"]
        n_actual = A_default.shape[0]
        st.info(f"ℹ️ {sistemas_predef[sistema_predef]['desc']}")
    else:
        n_actual = n
        A_default = np.eye(n_actual)
        b_default = np.ones(n_actual)

    # Construir tabla editable para la matriz A
    A_input = []
    for i in range(n_actual):
        cols_fila = st.columns(n_actual)
        fila = []
        for j in range(n_actual):
            val_def = float(A_default[i, j]) if i < A_default.shape[0] and j < A_default.shape[1] else 0.0
            val = cols_fila[j].number_input(
                f"a[{i+1},{j+1}]", value=val_def, format="%.4f",
                key=f"a_{i}_{j}", label_visibility="collapsed"
            )
            fila.append(val)
        A_input.append(fila)

    # Etiquetas de columnas
    col_labels = st.columns(n_actual)
    for j in range(n_actual):
        col_labels[j].markdown(f"<center>x<sub>{j+1}</sub></center>", unsafe_allow_html=True)

with col_b:
    st.markdown("**Vector b**")
    b_input = []
    for i in range(n_actual):
        val_def = float(b_default[i]) if i < len(b_default) else 0.0
        val = st.number_input(
            f"b[{i+1}]", value=val_def, format="%.4f",
            key=f"b_{i}", label_visibility="collapsed"
        )
        b_input.append(val)

# Convertir a arrays numpy
A = np.array(A_input, dtype=float)
b = np.array(b_input, dtype=float)


st.divider()
st.subheader(" Análisis de la Matriz")

col1, col2, col3, col4 = st.columns(4)

det_A = np.linalg.det(A)
cond_A = numero_condicion(A)
dd = es_diagonal_dominante(A)
rango_A = np.linalg.matrix_rank(A)

with col1:
    st.metric("det(A)", f"{det_A:.4f}")
with col2:
    color_cond = "normal" if cond_A < 1e6 else "inverse"
    st.metric("κ(A) — Núm. condición", f"{cond_A:.2e}")
with col3:
    st.metric("rango(A)", f"{rango_A} / {n_actual}")
with col4:
    st.metric("Diag. dominante", "Sí " if dd else "No ")

if cond_A > 1e6:
    st.markdown('<div class="advertencia"><b>Advertencia:</b> El número de condición es muy alto (κ(A) > 10⁶). El sistema está mal condicionado — los métodos iterativos pueden diverger o converger muy lentamente.</div>', unsafe_allow_html=True)

if abs(det_A) < 1e-12:
    st.markdown('<div class="error-box"> <b>Error:</b> det(A) ≈ 0. El sistema puede ser singular o no tener solución única.</div>', unsafe_allow_html=True)


st.divider()

if st.button(" Resolver Sistema", use_container_width=True, type="primary"):

    resultados_finales = {}
    x_exacto = None

    try:
        x_exacto = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        st.error("El sistema es singular. No tiene solución única.")
        st.stop()

    # ── Seleccionar métodos a ejecutar ──
    if metodo_sel == "Todos":
        metodos_a_ejecutar = ["Factorización LU", "Jacobi", "Gauss-Seidel",
                              f"SOR (ω={omega:.2f})", "PCG-Jacobi"]
    else:
        metodos_a_ejecutar = [metodo_sel]

    # ── Ejecutar métodos ──
    progress_bar = st.progress(0)
    for idx, met in enumerate(metodos_a_ejecutar):
        progress_bar.progress((idx + 1) / len(metodos_a_ejecutar))
        try:
            if met == "Factorización LU":
                x, info = factorizacion_lu(A, b)
            elif met == "Jacobi":
                x, info = jacobi(A, b, tol=tol, max_iter=int(max_iter))
            elif met == "Gauss-Seidel":
                x, info = gauss_seidel(A, b, tol=tol, max_iter=int(max_iter))
            elif met.startswith("SOR"):
                x, info = sor(A, b, omega=omega, tol=tol, max_iter=int(max_iter))
            elif met == "PCG-Jacobi":
                x, info = gradiente_conjugado_precondicionado(
                    A, b, tol=tol, max_iter=int(max_iter),
                    precondicionador=precond_pcg
                )
            else:
                continue
            info["error_vs_exacto"] = error_relativo(x, x_exacto)
            resultados_finales[met] = {"x": x, "info": info}
        except Exception as e:
            resultados_finales[met] = {"x": None, "info": {"error_msg": str(e), "convergio": False, "iteraciones": 0}}

    progress_bar.empty()

  
    st.subheader(" Resultados")

    # Solución exacta
    st.markdown("**Solución exacta (numpy.linalg.solve):**")
    cols_sol = st.columns(n_actual)
    for i, v in enumerate(x_exacto):
        cols_sol[i].metric(f"x{i+1}", f"{v:.8f}")

    st.divider()

    # Resultados por método
    for met, res in resultados_finales.items():
        with st.expander(f"{'' if res['info'].get('convergio', False) else ''} {met}", expanded=True):
            info = res["info"]

            if "error_msg" in info:
                st.error(f"Error: {info['error_msg']}")
                continue

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Iteraciones", info.get("iteraciones", "—"))
            col_r2.metric("Residuo ||Ax-b||", f"{info.get('residuo', float('nan')):.2e}")
            col_r3.metric("Error relativo", f"{info.get('error_vs_exacto', float('nan')):.2e}")
            conv_txt = " Sí" if info.get("convergio") else " No"
            col_r4.metric("Convergió", conv_txt)

            if res["x"] is not None:
                st.markdown("**Solución aproximada:**")
                cols_aprox = st.columns(n_actual)
                for i, v in enumerate(res["x"]):
                    diff = abs(v - x_exacto[i])
                    cols_aprox[i].metric(f"x{i+1}", f"{v:.8f}", delta=f"Δ={diff:.1e}")


    st.divider()
    st.subheader(" Gráficas de Convergencia")

    metodos_iter = {k: v for k, v in resultados_finales.items()
                    if k != "Factorización LU" and v["info"].get("errores")}

    if metodos_iter:
        fig, ax = plt.subplots(figsize=(10, 5))
        colores = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6", "#F39C12"]
        marcadores = ["o", "s", "^", "D"]

        for idx, (met_nombre, res) in enumerate(metodos_iter.items()):
            errores = res["info"].get("errores", [])
            if not errores:
                continue
            iters = list(range(1, len(errores) + 1))
            ax.semilogy(
                iters, errores,
                label=f"{met_nombre} ({len(errores)} iter)",
                color=colores[idx % len(colores)],
                marker=marcadores[idx % len(marcadores)],
                markevery=max(1, len(iters) // 10),
                linewidth=2.0,
            )

        ax.axhline(y=tol, color="gray", linestyle="--", alpha=0.7, label=f"Tolerancia ε={tol:.1e}")
        ax.set_xlabel("Iteración", fontsize=12)
        ax.set_ylabel("Error (escala logarítmica)", fontsize=12)
        ax.set_title("Convergencia de Métodos Iterativos", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No hay métodos iterativos con historial de errores para graficar.")


    if n_actual == 3:
        st.divider()
        st.subheader(" Visualización 3D — Intersección de Planos")

        fig3d = plt.figure(figsize=(9, 7))
        ax3d = fig3d.add_subplot(111, projection="3d")

        rango = max(abs(x_exacto)) * 2.5 + 3
        u = np.linspace(-rango, rango, 30)
        v = np.linspace(-rango, rango, 30)
        U, V = np.meshgrid(u, v)
        colores_p = ["#3498DB", "#E74C3C", "#2ECC71"]

        for i in range(3):
            a1, a2, a3 = A[i, 0], A[i, 1], A[i, 2]
            bi = b[i]
            try:
                if abs(a3) > 1e-10:
                    W = (bi - a1 * U - a2 * V) / a3
                    ax3d.plot_surface(U, V, W, alpha=0.3, color=colores_p[i])
            except Exception:
                pass

        ax3d.scatter([x_exacto[0]], [x_exacto[1]], [x_exacto[2]],
                     color="gold", s=250, zorder=15,
                     label=f"Solución: ({x_exacto[0]:.3f}, {x_exacto[1]:.3f}, {x_exacto[2]:.3f})")
        ax3d.set_xlabel("x₁"); ax3d.set_ylabel("x₂"); ax3d.set_zlabel("x₃")
        ax3d.set_title("Tres planos y su punto de intersección", fontweight="bold")
        ax3d.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3d)
        plt.close(fig3d)

    
    st.divider()
    st.subheader(" Tabla Resumen Comparativa")

    filas = []
    for met, res in resultados_finales.items():
        info = res["info"]
        filas.append({
            "Método": met,
            "Iteraciones": info.get("iteraciones", "—"),
            "Residuo ||Ax-b||": f"{info.get('residuo', float('nan')):.2e}" if "residuo" in info else "—",
            "Error relativo": f"{info.get('error_vs_exacto', float('nan')):.2e}" if "error_vs_exacto" in info else "—",
            "Convergió": " Sí" if info.get("convergio") else " No",
        })

    import pandas as pd
    df_tabla = pd.DataFrame(filas)
    st.dataframe(df_tabla, use_container_width=True)


st.divider()
st.markdown("""
<center style='color:#888; font-size:0.85rem;'>
    Métodos Numéricos — Álgebra Lineal Aplicada &nbsp;|&nbsp;
    LU · Jacobi · Gauss-Seidel · SOR · PCG
</center>
""", unsafe_allow_html=True)
