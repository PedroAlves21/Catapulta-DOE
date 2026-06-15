"""
optimization.py
---------------
Responsável pela otimização dos parâmetros da catapulta utilizando scipy.optimize.
"""

from __future__ import annotations
from typing import TypeAlias, Any, cast

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, differential_evolution

# Limites do experimento (bounds)

Number: TypeAlias = int | float
Bounds: TypeAlias = list[tuple[Number, Number]]

BOUNDS: Bounds = [
    (156, 176),   # release_angle
    (90, 140),    # firing_angle
    (211, 290),   # cup_elevation
    (100, 200),   # pin_elevation
    (100, 200),    # bungee_elevation
]

def predict_distance(x, scaler, poly, theta: np.ndarray) -> float:
    """
    Prediz a distância usando o modelo polinomial ajustado.

    x precisa estar na ordem:
    [release_angle, firing_angle, cup_elevation, pin_elevation, bungee_elevation]
    """
    x = np.asarray(x, dtype=float).reshape(1, -1)

    if scaler is not None:
        x_processed = scaler.transform(x)
    else:
        x_processed = x

    phi = poly.transform(x_processed)

    y_pred = phi @ theta

    return float(y_pred[0])

def objective_function(x, target_distance: float, scaler, poly, theta: np.ndarray) -> float:
    """
    Função objetivo da otimização restrita.

    Minimiza a diferença entre a distância estimada pelo modelo e a distância alvo.

    objetivo = (g_hat(x) - d*)²
    """
    estimated_distance = predict_distance(x=x, scaler=scaler, poly=poly, theta=theta)
    error = estimated_distance - target_distance

    return error ** 2

# Otimização local

def optimize_local(target_distance: float, scaler, poly, theta: np.ndarray, bounds: Bounds = BOUNDS, initial_guess: list[float] | None = None):
    """
    Executa otimização local usando L-BFGS-B com bounds.
    """
    if initial_guess is None:
        initial_guess = [(lower + upper) / 2 for lower, upper in bounds]

    result = minimize(
        fun=objective_function,
        x0=np.asarray(initial_guess, dtype=float),
        args=(target_distance, scaler, poly, theta),
        bounds=bounds,
        method="L-BFGS-B",
    )

    predicted_distance = predict_distance(x=result.x, scaler=scaler, poly=poly, theta=theta)

    return {
        "method": "local_L-BFGS-B",
        "success": result.success,
        "message": result.message,
        "x": result.x,
        "target_distance": target_distance,
        "predicted_distance": predicted_distance,
        "absolute_error": abs(predicted_distance - target_distance),
        "objective_value": result.fun,
        "raw_result": result,
    }

# Otimização global

def optimize_global(target_distance: float, scaler, poly, theta: np.ndarray, bounds: Bounds = BOUNDS, seed: int = 42):
    """
    Executa otimização global usando differential_evolution.
    """
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(target_distance, scaler, poly, theta),
        rng=seed,
        polish=True,
        maxiter=1000,
        popsize=20,
        tol=1e-7,
    )

    predicted_distance = predict_distance(x=result.x, scaler=scaler, poly=poly, theta=theta)

    return {
        "method": "global_differential_evolution",
        "success": result.success,
        "message": result.message,
        "x": result.x,
        "target_distance": target_distance,
        "predicted_distance": predicted_distance,
        "absolute_error": abs(predicted_distance - target_distance),
        "objective_value": result.fun,
        "raw_result": result,
    }

def compare_optimization_methods(target_distance: float, scaler, poly, 
                                 theta: np.ndarray, bounds: Bounds = BOUNDS, initial_guess: list[float] | None = None) -> tuple[dict, dict]:
    """
    Executa otimização local e global e retorna os dois resultados.
    """
    local_result = optimize_local(target_distance=target_distance, scaler=scaler, poly=poly, theta=theta, bounds=bounds, initial_guess=initial_guess)
    global_result = optimize_global(target_distance=target_distance, scaler=scaler, poly=poly, theta=theta, bounds=bounds)

    return local_result, global_result

# Visualização 2D

def plot_2d_cut(target_distance: float, scaler, poly, 
                theta: np.ndarray, bounds: Bounds = BOUNDS, 
                fixed_x: list[float] | np.ndarray | None = None, variable_index: int = 1, variable_name: str = "firing_angle"):
    """
    Gera um corte 2D da distância estimada pelo modelo.

    Por padrão, varia firing_angle e mantém o resto fixo.
    """
    if fixed_x is None:
        fixed_x = [(lower + upper) / 2 for lower, upper in bounds]

    lower, upper = bounds[variable_index]
    values = np.linspace(lower, upper, 100)
    distances = []

    for value in values:
        x = fixed_x.copy()
        x[variable_index] = value

        predicted = predict_distance(x=x, scaler=scaler, poly=poly, theta=theta)
        distances.append(predicted)

    plt.figure(figsize=(8, 5))

    plt.plot(values, distances, label="Distância estimada")
    plt.axhline(target_distance, linestyle="--", color="red", label="Distância alvo")

    plt.xlabel(variable_name)
    plt.ylabel("Distância estimada")
    plt.title(f"Corte 2D da Função Estimada: {variable_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visuzalização 3D

def plot_3d_surface(scaler, poly, theta: np.ndarray, 
                    bounds: Bounds = BOUNDS, fixed_x: list[float] | np.ndarray | None = None, 
                    x_index: int = 0, y_index: int = 1, x_name: str = "release_angle", 
                    y_name: str = "firing_angle", highlight_x: list[float] | np.ndarray | None = None, 
                    highlight_predicted_distance: float | None = None, target_distance: float | None = None):
    """
    Gera superfície 3D da distância estimada pelo modelo.

    Se highlight_x for informado, plota um ponto destacando a solução.
    """

    if fixed_x is None:
        fixed_x = [(lower + upper) / 2 for lower, upper in bounds]

    x_values = np.linspace(bounds[x_index][0], bounds[x_index][1], 50)
    y_values = np.linspace(bounds[y_index][0], bounds[y_index][1], 50)

    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_config = fixed_x.copy()
            x_config[x_index] = X[i, j]
            x_config[y_index] = Y[i, j]

            Z[i, j] = predict_distance(x=x_config, scaler=scaler, poly=poly, theta=theta)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.75)

    if highlight_x is not None and highlight_predicted_distance is not None:
        highlight_x = np.asarray(highlight_x, dtype=float)

        x_plot = highlight_x[x_index]
        y_plot = highlight_x[y_index]
        z_plot = highlight_predicted_distance

        cast(Any, ax).scatter(x_plot, y_plot, z_plot, color="red", s=120, edgecolors="black", linewidths=1.0, depthshade=False, label="Solução otimizada")

        if target_distance is not None:
            cast(Any, ax).scatter([x_plot], [y_plot], [target_distance], color="orange", s=120, edgecolors="black", linewidths=1.0, depthshade=False, label="Distância alvo")
            ax.plot([x_plot, x_plot], [y_plot, y_plot], [target_distance, z_plot], color="black", linestyle="--")

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel("Distância estimada")
    ax.set_title("Superfície 3D da Função Estimada")
    ax.legend()

    plt.tight_layout()
    plt.show()
