FROM prefecthq/prefect:3.2.11-python3.12

# Instalar pacotes necessários e AWS CLI numa única camada
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip less groff ca-certificates && \
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Define o diretório de trabalho
WORKDIR /opt/prefect

# Copia requirements e instala com cache ativado
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o script do fluxo
COPY train.py .

# Comando padrão
CMD ["prefect", "version"]
