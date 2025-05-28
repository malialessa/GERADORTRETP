# Usa uma imagem base Python oficial leve
FROM python:3.9-slim-buster

# Define o diretório de trabalho na imagem
WORKDIR /app

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia todo o código da sua aplicação para o diretório de trabalho
COPY . .

# Define a porta que o contêiner deve escutar.
ENV PORT 8080

# Comando para iniciar sua aplicação FastAPI usando uvicorn.
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
