import pandas as pd

# Carrega o binário (exige paciência no parsing inicial)
df = pd.read_excel('queue_picking.xlsx', engine='openpyxl')

# Exporta para texto plano
# index=False evita a criação de uma coluna de índices redundante
df.to_csv('queue_picking.csv', index=False, encoding='utf-8')
print("Conversão concluída: queue_picking.csv gerado.")