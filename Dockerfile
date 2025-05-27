# Use uma imagem base Python oficial (versão 3.9 recomendada para compatibilidade)
# 'slim-buster' é uma versão menor e mais segura do Debian para Python, ideal para contêineres.
FROM python:3.9-slim-buster

# Define o diretório de trabalho dentro do contêiner.
# Todos os comandos subsequentes (COPY, RUN, CMD) serão executados a partir deste diretório.
WORKDIR /app

# Copia o arquivo requirements.txt do seu repositório para o diretório de trabalho do contêiner.
COPY requirements.txt .

# Instala todas as dependências listadas em requirements.txt.
# --no-cache-dir: não armazena arquivos temporários de download e build, o que reduz o tamanho da imagem.
# -r requirements.txt: instala os pacotes listados no arquivo.
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o restante do código da sua aplicação (incluindo main.py)
# da sua máquina (ou do repositório, no caso do Cloud Build) para o diretório de trabalho do contêiner.
COPY . .

# Define a variável de ambiente PORT para 8080.
# O Cloud Run espera que sua aplicação ouça requisições na porta especificada por essa variável de ambiente.
ENV PORT 8080

# Comando para iniciar a aplicação quando o contêiner for executado.
# "uvicorn" é o servidor ASGI que roda o seu aplicativo FastAPI.
# "main:app": 'main' refere-se ao arquivo main.py, e 'app' é a instância do FastAPI dentro dele.
# "--host 0.0.0.0": instrui o uvicorn a ouvir em todas as interfaces de rede dentro do contêiner.
# "--port 8080": instrui o uvicorn a ouvir na porta 8080, que é a porta que o Cloud Run direciona as requisições.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
