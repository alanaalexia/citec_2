1. Criar o ambiente virtual
Execute o comando abaixo para criar uma máquina virtual (venv):
python -m venv venv

2. Ativar o ambiente virtual
Ative o ambiente conforme o seu sistema operacional:
Linux / macOS: source venv/bin/activate
Windows: .\venv\Scripts\activate

3. Instalar dependências
Com o ambiente ativado, faça upgrade de pip e instale os pacotes listados no arquivo requirements.txt:
pip install --upgrade pip
pip install -r requirements.txt