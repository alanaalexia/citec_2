import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Carregamento ──────────────────────────────────────────────────────────────
df = pd.read_csv("picking_tratado.csv")
df["DATE_FINISHED"] = pd.to_datetime(df["DATE_FINISHED"])

# ── Filtro: apenas blocos com 1 item único ────────────────────────────────────
df_single = df[~df["itens"].str.contains(",")].copy()
df_single["ITEM"] = df_single["itens"].str.strip()

print(f"Blocos totais      : {len(df)}")
print(f"Blocos item único  : {len(df_single)} ({len(df_single)/len(df)*100:.1f}%)\n")

# ── Filtro: apenas itens com observações suficientes para regressão ───────────
MIN_OBS = 10
contagem = df_single["ITEM"].value_counts()
itens_validos = contagem[contagem >= MIN_OBS].index
df_single = df_single[df_single["ITEM"].isin(itens_validos)].copy()

print(f"Itens com >= {MIN_OBS} obs : {len(itens_validos)}")
print(f"Obs restantes      : {len(df_single)}\n")

# ── Regressão por item: delta_tempo ~ total_done ──────────────────────────────
resultados = []

for item, grupo in df_single.groupby("ITEM"):
    X = grupo["total_done"].values.reshape(-1, 1)
    y = grupo["delta_tempo"].values

    if X.std() == 0:          # sem variação em total_done → skip
        continue

    modelo = LinearRegression().fit(X, y)
    r2     = r2_score(y, modelo.predict(X))

    resultados.append({
        "ITEM"        : item,
        "n_obs"       : len(grupo),
        "intercepto"  : modelo.intercept_,
        "coef"        : modelo.coef_[0],   # s por peça adicional
        "r2"          : r2,
        "total_done_mean": grupo["total_done"].mean(),
        "delta_tempo_mean": grupo["delta_tempo"].mean(),
    })

res = pd.DataFrame(resultados).sort_values("coef", ascending=False).reset_index(drop=True)

print("── Resultados por item (ordenado por coeficiente) ───────────────────────")
print(res.to_string(index=False))
res.to_csv("reg_itens_resultado.csv", index=False)
print("\nSalvo: reg_itens_resultado.csv\n")

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1 — Coeficiente (s/peça) por item, com tamanho = n_obs
# ════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, max(5, len(res) * 0.45)))

cores = ["crimson" if c > 0 else "steelblue" for c in res["coef"]]
bars  = ax.barh(res["ITEM"].astype(str), res["coef"], color=cores, alpha=0.8)

# Anotação: n_obs em cada barra
for bar, n in zip(bars, res["n_obs"]):
    ax.text(
        bar.get_width() + res["coef"].abs().max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"n={n}", va="center", ha="left", fontsize=8, color="gray"
    )

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Coeficiente (segundos por peça adicional)")
ax.set_title("Efeito do volume de peças no delta_tempo — por item\n"
             "(vermelho = mais peças → mais tempo | azul = mais peças → menos tempo)")
plt.tight_layout()
plt.savefig("reg_itens_coeficientes.png", dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 2 — Scatter por item com linha de regressão individual
# Mostra os top 12 itens com maior |coef| para não poluir
# ════════════════════════════════════════════════════════════════════════════
top_itens = res.reindex(res["coef"].abs().sort_values(ascending=False).index).head(12)["ITEM"].tolist()
df_top    = df_single[df_single["ITEM"].isin(top_itens)].copy()

n_cols = 3
n_rows = int(np.ceil(len(top_itens) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
axes = axes.flatten()

for ax, item in zip(axes, top_itens):
    grupo = df_top[df_top["ITEM"] == item]
    X = grupo["total_done"].values.reshape(-1, 1)
    y = grupo["delta_tempo"].values

    modelo = LinearRegression().fit(X, y)
    r2     = r2_score(y, modelo.predict(X))
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    cor = "crimson" if modelo.coef_[0] > 0 else "steelblue"
    ax.scatter(grupo["total_done"], grupo["delta_tempo"],
               alpha=0.4, s=15, color=cor)
    ax.plot(x_line, modelo.predict(x_line), color=cor, linewidth=2)
    ax.set_title(f"Item {item}\ncoef={modelo.coef_[0]:.2f} s/peça  R²={r2:.3f}  n={len(grupo)}",
                 fontsize=9)
    ax.set_xlabel("total_done", fontsize=8)
    ax.set_ylabel("delta_tempo (s)", fontsize=8)

# Oculta eixos extras
for ax in axes[len(top_itens):]:
    ax.set_visible(False)

plt.suptitle("Top 12 itens — Regressão delta_tempo ~ total_done", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("reg_itens_scatter_top12.png", dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 3 — R² por item (qualidade do ajuste)
# ════════════════════════════════════════════════════════════════════════════
res_sorted_r2 = res.sort_values("r2", ascending=True)

fig, ax = plt.subplots(figsize=(8, max(4, len(res) * 0.4)))
ax.barh(res_sorted_r2["ITEM"].astype(str), res_sorted_r2["r2"],
        color="mediumpurple", alpha=0.8)
ax.axvline(0.3, color="orange", linestyle="--", linewidth=1, label="R²=0.30")
ax.set_xlabel("R²")
ax.set_title("Qualidade da regressão por item\n(quanto total_done explica delta_tempo)")
ax.legend()
plt.tight_layout()
plt.savefig("reg_itens_r2.png", dpi=150)
plt.show()