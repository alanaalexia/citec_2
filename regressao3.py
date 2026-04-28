import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import statsmodels.formula.api as smf

sns.set_theme(style="whitegrid")

# ── Carregamento ──────────────────────────────────────────────────────────────
df = pd.read_csv("picking_tratado.csv")
df["DATE_FINISHED"] = pd.to_datetime(df["DATE_FINISHED"])

df = df.rename(columns={
    "USER_FINISHED": "usuario",
    "total_done":    "pecas",
    "delta_tempo":   "delta",
})

# ── Filtro: funcionários com >= 100 operações ─────────────────────────────────
min_operacoes = 100
contagem = df["usuario"].value_counts()
funcionarios_validos = contagem[contagem >= min_operacoes].index
df_filtrado = df[df["usuario"].isin(funcionarios_validos)].copy()

print(f"Funcionários com >= {min_operacoes} operações: {len(funcionarios_validos)}")
print(f"Registros utilizados: {len(df_filtrado)} de {len(df)}")

# ── Amostra estratificada (até 300 por funcionário) ───────────────────────────
amostra = pd.concat([
    grp.sample(min(len(grp), 300), random_state=42)
    for _, grp in df_filtrado.groupby("usuario")
]).reset_index(drop=True)

print(f"Tamanho da amostra: {len(amostra)}")

# ── OLS com efeitos fixos por funcionário ─────────────────────────────────────
modelo3 = smf.ols("delta ~ pecas + C(usuario)", data=amostra).fit()

# ── Extrai efeitos fixos ──────────────────────────────────────────────────────
efeitos = (
    modelo3.params[modelo3.params.index.str.startswith("C(usuario)")]
           .rename(lambda x: x.replace("C(usuario)[T.", "").replace("]", ""))
)

# Adiciona o funcionário de referência (efeito = 0 no OLS) antes de centralizar
efeitos["referencia"] = 0.0

# Centraliza em torno da média do grupo e ordena
efeitos = (efeitos - efeitos.mean()).sort_values()

# ── Gráfico ───────────────────────────────────────────────────────────────────
n = len(efeitos)
fig, ax = plt.subplots(figsize=(12, max(6, n * 0.35)))

cores = ["steelblue" if v < 0 else "crimson" for v in efeitos]
bars  = ax.barh(range(n), efeitos.values, color=cores, edgecolor="white", linewidth=0.4)

ax.set_yticks(range(n))
ax.set_yticklabels(efeitos.index, fontsize=7)
ax.axvline(0, color="black", linewidth=1, linestyle="--")

for i, (val, _) in enumerate(zip(efeitos.values, bars)):
    ax.text(
        val + (5 if val >= 0 else -5),
        i, f"{val:+.0f}s",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=6,
    )

ax.set_xlabel(
    "Desvio em relação à média do grupo (s)\n← mais rápido que a média   |   mais lento que a média →",
    fontsize=10,
)
ax.set_title(
    "Regressão 3 — Velocidade relativa por funcionário\n(controlando por volume de peças, mín. 100 operações)",
    fontsize=11, pad=12,
)
ax.legend(
    handles=[Patch(color="steelblue", label="Mais rápido que a média"),
             Patch(color="crimson",   label="Mais lento que a média")],
    loc="lower right", fontsize=9,
)

plt.tight_layout()
plt.savefig("reg3_efeitos_funcionarios.png", dpi=180, bbox_inches="tight")
plt.show()

# ── Resumo ────────────────────────────────────────────────────────────────────
print(f"\n── Regressão 3 ──────────────────────────────────────")
print(f"  R²  : {modelo3.rsquared:.4f}")
print(f"  N   : {int(modelo3.nobs)}")
print(f"\n  5 mais rápidos que a média:")
print(efeitos.head(5).to_string())
print(f"\n  5 mais lentos que a média:")
print(efeitos.tail(5).to_string())