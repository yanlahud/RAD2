
# Base image com suporte para Python e compatível com MONAI
FROM python:3.10-slim

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho
WORKDIR /app

# Copiar dependências
COPY requirements.txt .

# Instalar dependências
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar todo o projeto
COPY . .

# Expor a porta padrão do Uvicorn
EXPOSE 8000

# Comando para iniciar o app FastAPI com hot reload (produção usar --no-reload)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
