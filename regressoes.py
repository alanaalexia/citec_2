import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf

# ── Carregamento ──────────────────────────────────────────────────────────────
df = pd.read_csv("picking_tratado.csv")
df["DATE_FINISHED"] = pd.to_datetime(df["DATE_FINISHED"])
df["date_ordinal"]  = df["DATE_FINISHED"].map(pd.Timestamp.toordinal)

sns.set_theme(style="whitegrid")

# ════════════════════════════════════════════════════════════════════════════
# REGRESSÃO 1 — delta_tempo ~ total_done  (operação individual)
# Pergunta: operações com mais peças demoram proporcionalmente mais?
# ════════════════════════════════════════════════════════════════════════════
X1 = df["total_done"].values.reshape(-1, 1)
y1 = df["delta_tempo"].values
modelo1 = LinearRegression().fit(X1, y1)
r2_1    = r2_score(y1, modelo1.predict(X1))

x_line1 = np.linspace(X1.min(), X1.max(), 200).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["total_done"], df["delta_tempo"], alpha=0.1, s=5, color="steelblue", label="Observações")
ax.plot(x_line1, modelo1.predict(x_line1), color="crimson", linewidth=2,
        label=f"Regressão linear (R²={r2_1:.3f})")
ax.set_xlabel("Peças movidas (total_done)")
ax.set_ylabel("Delta tempo (s)")
ax.set_title("Regressão 1 — Tempo por operação vs. Peças movidas")
ax.legend()
plt.tight_layout()
plt.savefig("reg1_tempo_vs_pecas.png", dpi=150)
plt.show()

print("── Regressão 1 ──────────────────────────────────────")
print(f"  Intercepto  : {modelo1.intercept_:.2f} s")
print(f"  Coeficiente : {modelo1.coef_[0]:.4f} s / peça")
print(f"  R²          : {r2_1:.4f}\n")

# ════════════════════════════════════════════════════════════════════════════
# REGRESSÃO 2 — produtividade ao longo do tempo  (agregado por funcionário×dia)
# Pergunta: os funcionários estão ficando mais rápidos com o tempo?
# Métrica: peças por segundo = total_done / delta_tempo
# ════════════════════════════════════════════════════════════════════════════
diario = (
    df.groupby(["USER_FINISHED", "DATE_FINISHED", "date_ordinal"])
      .agg(total_done=("total_done", "sum"), delta_tempo=("delta_tempo", "sum"))
      .reset_index()
)
diario["produtividade"] = diario["total_done"] / diario["delta_tempo"]

X2     = diario["date_ordinal"].values.reshape(-1, 1)
y2     = diario["produtividade"].values
modelo2 = LinearRegression().fit(X2, y2)
r2_2    = r2_score(y2, modelo2.predict(X2))

x_line2  = np.linspace(X2.min(), X2.max(), 200).reshape(-1, 1)
datas_line = [pd.Timestamp.fromordinal(int(x)) for x in x_line2.flatten()]

fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(diario["DATE_FINISHED"], diario["produtividade"],
           alpha=0.3, s=10, color="steelblue", label="Funcionário×dia")
ax.plot(datas_line, modelo2.predict(x_line2), color="crimson", linewidth=2,
        label=f"Tendência (R²={r2_2:.3f})")
ax.set_xlabel("Data")
ax.set_ylabel("Produtividade (peças / s)")
ax.set_title("Regressão 2 — Produtividade agregada ao longo do tempo")
ax.legend()
plt.tight_layout()
plt.savefig("reg2_produtividade_tempo.png", dpi=150)
plt.show()

print("── Regressão 2 ──────────────────────────────────────")
print(f"  Tendência diária : {modelo2.coef_[0] * 86400:+.6f} peças/s por dia")
print(f"  R²               : {r2_2:.4f}\n")

# ════════════════════════════════════════════════════════════════════════════
# REGRESSÃO 3 — efeitos fixos por funcionário  (OLS com dummies)
# Pergunta: quem é intrinsecamente mais rápido, controlando pelo volume?
# Modelo: delta_tempo ~ total_done + C(USER_FINISHED)
# ════════════════════════════════════════════════════════════════════════════
# ── Regressão 3 corrigida ─────────────────────────────────────────────────────

# Filtro: apenas funcionários com pelo menos 100 operações
min_operacoes = 100
contagem = df["USER_FINISHED"].value_counts()
funcionarios_validos = contagem[contagem >= min_operacoes].index
df_filtrado = df[df["USER_FINISHED"].isin(funcionarios_validos)].copy()

print(f"Funcionários com >= {min_operacoes} operações: {len(funcionarios_validos)}")
print(f"Registros utilizados: {len(df_filtrado)} de {len(df)}")

# Amostra estratificada — garante representação proporcional por funcionário
amostra = (
    df_filtrado.groupby("USER_FINISHED", group_keys=False)
               .apply(lambda x: x.sample(min(len(x), 300), random_state=42))
)

modelo3 = smf.ols("delta_tempo ~ total_done + C(USER_FINISHED)", data=amostra).fit()

# Extrai efeitos fixos
efeitos = (
    modelo3.params[modelo3.params.index.str.startswith("C(USER_FINISHED)")]
           .rename(lambda x: x.replace("C(USER_FINISHED)[T.", "").replace("]", ""))
           .sort_values()
)

# ── Gráfico corrigido ─────────────────────────────────────────────────────────
n = len(efeitos)
altura_por_barra = 0.35
fig, ax = plt.subplots(figsize=(12, max(6, n * altura_por_barra)))

cores = ["steelblue" if v < 0 else "crimson" for v in efeitos]
bars = ax.barh(range(n), efeitos.values, color=cores, edgecolor="white", linewidth=0.4)

# Nomes no eixo Y com fonte menor
ax.set_yticks(range(n))
ax.set_yticklabels(efeitos.index, fontsize=7)

# Linha de referência
ax.axvline(0, color="black", linewidth=1, linestyle="--")

# Valor no final de cada barra
for i, (val, bar) in enumerate(zip(efeitos.values, bars)):
    ax.text(
        val + (5 if val >= 0 else -5),
        i,
        f"{val:+.0f}s",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=6,
        color="black",
    )

ax.set_xlabel("Efeito sobre delta_tempo (s)\n← mais rápido que referência   |   mais lento →", fontsize=10)
ax.set_title(
    "Regressão 3 — Velocidade relativa por funcionário\n(controlando por volume de peças, mín. 100 operações)",
    fontsize=11,
    pad=12,
)

# Legenda manual
from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(color="steelblue", label="Mais rápido"), Patch(color="crimson", label="Mais lento")],
    loc="lower right", fontsize=9,
)

plt.tight_layout()
plt.savefig("reg3_efeitos_funcionarios.png", dpi=180, bbox_inches="tight")
plt.show()

print(f"\n── Regressão 3 ──────────────────────────────────────")
print(f"  R²  : {modelo3.rsquared:.4f}")
print(f"  N   : {int(modelo3.nobs)}")
print(f"\n  5 mais rápidos:")
print(efeitos.head(5).to_string())
print(f"\n  5 mais lentos:")
print(efeitos.tail(5).to_string())