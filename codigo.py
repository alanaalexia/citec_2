import pandas as pd

# ── 1. Carregamento ──────────────────────────────────────────────────────────
df = pd.read_csv("picking.csv")
df.columns = df.columns.str.strip().str.replace(r"^\.", "", regex=True)

df = df[(df["STATUS"] == "Finished") & (df["TYPE_MOVE"] == "Picking")].copy()

# ── 2. Parse de datas/horas ──────────────────────────────────────────────────
df["DATE_FINISHED"] = pd.to_datetime(df["DATE_FINISHED"], format='mixed').dt.date
df["TIME_FINISHED"] = pd.to_datetime(df["TIME_FINISHED"])
df["time_seconds"] = (
    df["TIME_FINISHED"].dt.hour * 3600
    + df["TIME_FINISHED"].dt.minute * 60
    + df["TIME_FINISHED"].dt.second
)

# ── 3. Ordenação ─────────────────────────────────────────────────────────────
df = df.sort_values(["USER_FINISHED", "DATE_FINISHED", "time_seconds"]).reset_index(drop=True)

# ── 4. Agrupamento de atividades "simultâneas" (gap ≤ 30 s) ─────────────────
THRESHOLD = 30

def assign_group(subdf):
    group_ids = []
    gid = 0
    prev_t = None
    for t in subdf["time_seconds"]:
        if prev_t is None or (t - prev_t) > THRESHOLD:
            gid += 1
        group_ids.append(gid)
        prev_t = t
    subdf = subdf.copy()
    subdf["group_id"] = group_ids
    return subdf

# Guarda colunas antes do apply (workaround bug pandas)
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

# ── 5. Representante de cada grupo ───────────────────────────────────────────
group_keys = ["USER_FINISHED", "DATE_FINISHED", "group_id"]

group_stats = df.groupby(group_keys).agg(
    t_start=("time_seconds", "min"),
    t_end=("time_seconds", "max"),
    total_done=("_DONE", "sum"),
    n_items=("ITEM", "count"),
).reset_index()  # USER_FINISHED, DATE_FINISHED voltam como colunas normais aqui

# ── 6. Delta entre grupos consecutivos — SEM apply ───────────────────────────
group_stats = group_stats.sort_values(
    ["USER_FINISHED", "DATE_FINISHED", "group_id"]
).reset_index(drop=True)

# shift vetorizado dentro de cada funcionário×dia
group_stats["prev_t_end"] = (
    group_stats.groupby(["USER_FINISHED", "DATE_FINISHED"])["t_end"]
               .shift(1)
)

group_stats["delta_tempo"]   = group_stats["t_start"] - group_stats["prev_t_end"]
group_stats["duracao_bloco"] = group_stats["t_end"]   - group_stats["t_start"]

# Descarta primeiro registro de cada funcionário×dia (prev_t_end == NaN)
group_stats = group_stats.dropna(subset=["delta_tempo"]).reset_index(drop=True)

# ── 7. Resultado final ────────────────────────────────────────────────────────
result = group_stats[[
    "USER_FINISHED", "DATE_FINISHED", "group_id",
    "n_items", "total_done",
    "t_start", "t_end", "duracao_bloco", "delta_tempo",
]].copy()

print(result.head(20).to_string())
print(f"\nTotal de blocos após descarte: {len(result)}")
print(f"\nEstatísticas de delta_tempo (segundos):")
print(result["delta_tempo"].describe())