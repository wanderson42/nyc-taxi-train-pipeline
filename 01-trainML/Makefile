# Cetifique-se de ter executado:
# minikube start 
# eval $(minikube docker-env)

# Variáveis
YEAR ?= 2024
TAXI_TYPE ?= green
BUCKET = mlflow-cloud-artifacts
DATASET_FOLDER = nyc-trip-data
ARTIFACT_FOLDER = nyc-mlflow-artifacts
ENTRYPOINT = train.py:main_flow
DEPLOYMENT_NAME = train-nyc
TIMEZONE = America/Sao_Paulo
POOL = minikube-workpool
IMAGE_NAME ?= wlf42/nyc-taxi-flow:latest  

# Função para gerar o prefect.yaml
generate-prefect-yaml:
	@echo "Gerando o arquivo prefect.yaml com o script Python..."
	@python3 generate_prefect_yaml.py

# Função para fazer o deploy para os meses
deploy-months: generate-prefect-yaml
	@for month in $$(seq -w 01 06); do \
		EXP_NAME=nyc-$(TAXI_TYPE)-taxi-$(YEAR)-$$month; \
		DATE=$(YEAR)-$$month-01; \
		CRON="0 9 5 $$month *"; \
		echo "🔧 Deploying for month $$month ($$DATE)..."; \
		echo "y" | prefect deploy $(ENTRYPOINT) \
			--name $(DEPLOYMENT_NAME)-$$month \
			--pool $(POOL) \
			--cron "$$CRON" \
			--timezone $(TIMEZONE) \
			--params "{\"exp_name\": \"$$EXP_NAME\", \
			           \"taxi_type\": \"$(TAXI_TYPE)\", \
			           \"date\": \"$$DATE\", \
			           \"s3bucket_name\": \"$(BUCKET)\", \
			           \"s3bucket_dataset_folder\": \"$(DATASET_FOLDER)\", \
			           \"s3bucket_mlflow_artifact_folder\": \"$(ARTIFACT_FOLDER)\"}" \
			--description "Deploy mensal do fluxo $(DEPLOYMENT_NAME) para o mês de $$month/$(YEAR)" \
			--job-variable image=$(IMAGE_NAME) ; \
	done
