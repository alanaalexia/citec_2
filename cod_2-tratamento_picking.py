import pandas as pd

# ── 1. Carregamento e limpeza inicial ────────────────────────────────────────
df = pd.read_csv("picking.csv")
df.columns = df.columns.str.strip().str.replace(r"^\.", "", regex=True)
df = df[(df["STATUS"] == "Finished") & (df["TYPE_MOVE"] == "Picking")].copy()

# ── 2. Parse de datas e horários ─────────────────────────────────────────────
df["DATE_FINISHED"] = pd.to_datetime(df["DATE_FINISHED"], format="mixed").dt.date
df["TIME_FINISHED"] = pd.to_datetime(df["TIME_FINISHED"])
df["time_seconds"]  = (
    df["TIME_FINISHED"].dt.hour   * 3600
    + df["TIME_FINISHED"].dt.minute * 60
    + df["TIME_FINISHED"].dt.second
)

# ── 3. Ordenação ─────────────────────────────────────────────────────────────
df = df.sort_values(["USER_FINISHED", "DATE_FINISHED", "time_seconds"]).reset_index(drop=True)

# ── 4. Agrupamento de atividades simultâneas (gap ≤ 30 s) ───────────────────
THRESHOLD = 30

def assign_group(subdf):
    """Atribui um group_id sequencial: novo grupo sempre que gap > THRESHOLD."""
    group_ids, gid, prev_t = [], 0, None
    for t in subdf["time_seconds"]:
        if prev_t is None or (t - prev_t) > THRESHOLD:
            gid += 1
        group_ids.append(gid)
        prev_t = t
    subdf = subdf.copy()
    subdf["group_id"] = group_ids
    return subdf

# Workaround: pandas pode promover colunas de agrupamento ao índice no apply
user_col = df["USER_FINISHED"].values
date_col = df["DATE_FINISHED"].values

df = (
    df.groupby(["USER_FINISHED", "DATE_FINISHED"], group_keys=False)
      .apply(assign_group)
      .reset_index(drop=True)
)

if "USER_FINISHED" not in df.columns:
    df["USER_FINISHED"] = user_col
if "DATE_FINISHED" not in df.columns:
    df["DATE_FINISHED"] = date_col

# ── 5. Agregação por grupo ────────────────────────────────────────────────────
group_keys = ["USER_FINISHED", "DATE_FINISHED", "group_id"]

result = (
    df.groupby(group_keys)
      .agg(
          t_start    = ("time_seconds", "min"),
          t_end      = ("time_seconds", "max"),
          total_done = ("_DONE",        "sum"),
          n_items    = ("ITEM",         "count"),
      )
      .reset_index()
      .sort_values(group_keys)
      .reset_index(drop=True)
)

# ── 6. Delta de tempo entre grupos consecutivos ───────────────────────────────
result["prev_t_end"]    = result.groupby(["USER_FINISHED", "DATE_FINISHED"])["t_end"].shift(1)
result["delta_tempo"]   = result["t_start"] - result["prev_t_end"]
result["duracao_bloco"] = result["t_end"]   - result["t_start"]

# Descarta o primeiro registro de cada funcionário×dia (sem predecessor)
result = result.dropna(subset=["delta_tempo"]).reset_index(drop=True)

# ── 7. Remoção de outliers ────────────────────────────────────────────────────
# Deltas > 1h indicam pausas/almoço/troca de turno, não picking real
LIMITE_OUTLIER = 3600  # segundos
result = result[result["delta_tempo"] <= LIMITE_OUTLIER].reset_index(drop=True)

# ── 8. Chave legível e seleção final de colunas ───────────────────────────────
result["bloco_id"] = (
    result["USER_FINISHED"].astype(str)
    + "_" + result["DATE_FINISHED"].astype(str)
    + "_G" + result["group_id"].astype(str).str.zfill(2)
)

result = result[[
    "bloco_id", "USER_FINISHED", "DATE_FINISHED",
    "n_items", "total_done", "duracao_bloco", "delta_tempo",
]]

# ── 9. Resumo ─────────────────────────────────────────────────────────────────
print(result.head(20).to_string())
print(f"\nTotal de observações: {len(result)}")
print(f"\nEstatísticas de delta_tempo (segundos):")
print(result["delta_tempo"].describe())

# ── 10. Exportação ────────────────────────────────────────────────────────────
result.to_csv("picking_tratado.csv", index=False)
print("Arquivo salvo: picking_tratado.csv")