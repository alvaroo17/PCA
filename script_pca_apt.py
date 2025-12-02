{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww28600\viewh18000\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import pandas as pd\
import matplotlib.pyplot as plt\
from scipy.optimize import minimize\
\
# ---------------------------------------------------\
# 1. CARGA DE DATOS \
# ---------------------------------------------------\
\
df = pd.read_csv("rentabilidades.csv")\
\
# Columnas de activos (20 acciones)\
asset_cols = ["ASML", "SAP", "BBVA", "RHM", "RMS", "STLAM", "IBE", "HNR1",\
              "ELISA", "BAYN", "ZAL", "AGN", "HEIO", "RXL", "SAF", "SAN",\
              "STM", "PAH3", "ALSTOM", "BNP"]  # Ejemplo, c\'e1mbialo\
\
# Benchmark y tipo sin riesgo\
bench_col = "BENCH"   # benchmark (p.ej. EuroStoxx, tu cartera, etc.)\
rf_col    = "RF"      # t\'edtulo sin riesgo del periodo (en la misma frecuencia)\
\
# Rentabilidades de activos riesgosos\
R = df[asset_cols]          # matriz T x N\
rf_series = df[rf_col]      # serie RF (T x 1)\
bench = df[bench_col]       # serie benchmark\
\
\
mu = R.mean()               # vector de medias (por periodo: mensual)\
cov = R.cov()               # matriz de covarianza\
rf = rf_series.mean()       # tipo sin riesgo medio del periodo\
\
\
\
# ---------------------------------------------------\
# 2. FUNCIONES AUXILIARES\
# ---------------------------------------------------\
def portfolio_stats(weights, mu, cov, rf=0.0):\
\
       weights: vector de pesos (N,)\
    mu:      vector de rentabilidades esperadas (N,)\
    cov:     matriz de covarianza (N x N)\
    rf:      tipo sin riesgo\
    """\
    weights = np.array(weights)\
    ret = np.dot(weights, mu)\
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))\
    sharpe = (ret - rf) / vol if vol > 0 else 0\
    return ret, vol, sharpe\
\
\
def min_variance_for_target_return(target_ret, mu, cov):\
  \
    (long-only, sum(weights)=1).\
    """\
    n = len(mu)\
    mu = np.array(mu)\
    cov = np.array(cov)\
\
    # Funci\'f3n objetivo: minimizar varianza\
    def obj(weights):\
        return np.dot(weights.T, np.dot(cov, weights))\
\
    # Restricci\'f3n 1: suma de pesos = 1\
    cons = [\{"type": "eq", "fun": lambda w: np.sum(w) - 1\},\
            # Restricci\'f3n 2: rentabilidad objetivo\
            \{"type": "eq", "fun": lambda w: np.dot(w, mu) - target_ret\}]\
\
    # Acotamos entre 0 y 1 (long-only)\
    bounds = tuple((0, 1) for _ in range(n))\
\
    # Punto inicial: equiponderado\
    x0 = np.ones(n) / n\
\
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)\
    return res.x  # pesos \'f3ptimos\
\
\
def tangency_portfolio(mu, cov, rf=0.0):\
  \
    n = len(mu)\
    mu = np.array(mu)\
    cov = np.array(cov)\
\
    def neg_sharpe(weights):\
        ret, vol, sharpe = portfolio_stats(weights, mu, cov, rf)\
        return -sharpe\
\
    cons = [\{"type": "eq", "fun": lambda w: np.sum(w) - 1\}]\
    bounds = tuple((0, 1) for _ in range(n))\
    x0 = np.ones(n) / n\
\
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)\
    return res.x  # pesos de la cartera de tangencia\
\
\
# ---------------------------------------------------\
# 3. FRONTERA EFICIENTE (SOLO ACTIVOS DE RIESGO)\
# ---------------------------------------------------\
\
target_returns = np.linspace(mu.min(), mu.max(), 50)\
\
frontier_vols = []\
frontier_rets = []\
\
for tr in target_returns:\
    w = min_variance_for_target_return(tr, mu, cov)\
    ret, vol, _ = portfolio_stats(w, mu, cov, rf=rf)\
    frontier_rets.append(ret)\
    frontier_vols.append(vol)\
\
frontier_rets = np.array(frontier_rets)\
frontier_vols = np.array(frontier_vols)\
\
\
# ---------------------------------------------------\
# 4. CARTERA DE TANGENCIA Y CML (CON T\'cdTULO SIN RIESGO)\
# ---------------------------------------------------\
w_tan = tangency_portfolio(mu, cov, rf=rf)\
ret_tan, vol_tan, sharpe_tan = portfolio_stats(w_tan, mu, cov, rf=rf)\
\
print("Sharpe de la cartera de tangencia:", sharpe_tan)\
print("Rentabilidad tangencia:", ret_tan)\
print("Volatilidad tangencia:", vol_tan)\
\
# L\'ednea del mercado de capitales (CML)\
# Elegimos un rango de volatilidades (desde 0 hasta algo m\'e1s que la tangencia)\
cml_vols = np.linspace(0, vol_tan * 1.5, 50)\
cml_rets = rf + sharpe_tan * cml_vols\
\
\
# ---------------------------------------------------\
# 5. PUNTO DEL BENCHMARK\
# ---------------------------------------------------\
bench_ret = bench.mean()\
bench_vol = bench.std()\
\
\
# ---------------------------------------------------\
# 6. GR\'c1FICO\
# ---------------------------------------------------\
plt.figure(figsize=(10, 6))\
\
# Puntos de las acciones individuales\
asset_vols = R.std()\
asset_rets = R.mean()\
plt.scatter(asset_vols, asset_rets, label="Acciones individuales", alpha=0.7)\
for name in asset_cols:\
    plt.annotate(name, (asset_vols[name], asset_rets[name]), fontsize=8, alpha=0.7)\
\
# Frontera eficiente (solo riesgo)\
plt.plot(frontier_vols, frontier_rets, label="Frontera eficiente (sin TSR)", linestyle="--")\
\
# CML\
plt.plot(cml_vols, cml_rets, label="CML (con TSR)", linewidth=2)\
\
# Cartera de tangencia\
plt.scatter([vol_tan], [ret_tan], color="red", marker="*", s=150, label="Cartera de tangencia")\
\
# Benchmark\
plt.scatter([bench_vol], [bench_ret], color="black", marker="D", label="Benchmark")\
\
plt.xlabel("Volatilidad (desv. t\'edpica)")\
plt.ylabel("Rentabilidad esperada")\
plt.title("Frontera eficiente de Markowitz con t\'edtulo sin riesgo")\
plt.legend()\
plt.grid(True)\
plt.tight_layout()\
plt.show()\
}