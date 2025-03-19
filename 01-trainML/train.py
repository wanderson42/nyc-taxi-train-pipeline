################################################################################
#                                  Preâmbulo                                   #
################################################################################
#------------------------------------Begin ------------------------------------#
import os
import time
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import logging
import warnings
import pandas as pd
import numpy as np

from prefect import task, flow
from prefect.logging import get_run_logger
from prefect.artifacts import create_table_artifact, create_link_artifact
from prefect_aws import AwsCredentials

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError

import pyarrow.parquet as pq
from io import BytesIO

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import mlflow.pyfunc


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor    


################################################################################
#                                  Preâmbulo                                   #
################################################################################
#--------------------------------------End-------------------------------------#





@task(log_prints=True)
def configure_mlflow(exp_name: str,
                     s3bucket_name : str,
                     s3bucket_mlflow_artifact_folder : str):
    """
    Configura o experimento no MLflow e inicia um novo run pai.

    Args:
        exp_name (str): Nome do experimento.

    Returns:
        str: ID do run pai iniciado.
    """
    logger = get_run_logger()

    # Configura a URI do servidor remoto MLflow
    TRACKING_SERVER_HOST = "http://54.233.120.110:5000"
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_SERVER_HOST
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
    logger.info(f"Tracking URI configurada para: '{mlflow.get_tracking_uri()}'")

    # Criar cliente MLflow
    client = MlflowClient()
    
    # Verificar se o experimento já existe
    experiment = client.get_experiment_by_name(exp_name)

    if experiment is not None and experiment.lifecycle_stage == "deleted":
        logger.warning(f"⚠️ O experimento '{exp_name}' está deletado. Restaurando...")
        client.restore_experiment(experiment.experiment_id)  # Restaura o experimento

        # Buscar novamente após restaurar
        experiment = client.get_experiment_by_name(exp_name)

    if experiment is None:
        # Criar um novo experimento apenas se ele não existir
        s3_key = f"s3://{s3bucket_name}/{s3bucket_mlflow_artifact_folder}"  # Caminho dentro do S3
        experiment_id = client.create_experiment(name=exp_name, artifact_location=s3_key)
        logger.info(f"🆕 Novo experimento criado com ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"✅ Experimento existente com ID: {experiment_id}")

    # **Importante**: Configurar explicitamente para evitar problemas de cache
    mlflow.set_experiment(exp_name)
 
    # Finalizar qualquer run ativo antes de iniciar um novo
    if mlflow.active_run():
        logger.info(f"🛑 Finalizando run ativo com ID: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    print()
 
    # Inicia um run pai (parent) para agrupar os experimentos
    parent_run = mlflow.start_run(experiment_id=experiment_id, nested= True)
    parent_run_id = parent_run.info.run_id
    logger.info(f"🚀 Iniciando um run pai no experimento '{exp_name}' (Parent ID: {parent_run_id}).")

    return parent_run_id


################################################################################
#                                Data Ingestion                                #
################################################################################
#------------------------------------Begin ------------------------------------#


@task(retries=5, retry_delay_seconds=3, log_prints=True)
async def aws_fetch_data(taxi_type: str,
                         date: str,
                         bucket_name: str,
                         s3bucket_dataset_folder: str):

    aws_credentials_block = await AwsCredentials.load("aws-s3-creds")
    session = aws_credentials_block.get_boto3_session()
    s3_client = session.client('s3')

    processed_date = datetime.strptime(date, "%Y-%m-%d")
    time.sleep(5)  # Simula tempo de espera

    train_date = processed_date
    test_date = processed_date + relativedelta(months=1)

    train_file = f"{taxi_type}_tripdata_{train_date.year}-{str(train_date.month).zfill(2)}.parquet"
    test_file = f"{taxi_type}_tripdata_{test_date.year}-{str(test_date.month).zfill(2)}.parquet"

    files_content = []

    for file in [train_file, test_file]:
        s3_key = f"{s3bucket_dataset_folder}/{file}"

        try:
            file_obj = BytesIO()
            s3_client.download_fileobj(bucket_name, s3_key, file_obj)
            file_obj.seek(0)

            print(f"Arquivo {file} carregado na memória com sucesso!")
            files_content.append(file_obj.read())

        except NoCredentialsError:
            print("Erro: Credenciais da AWS não encontradas.")
            raise
        except BotoCoreError as e:
            print(f"Erro ao baixar o arquivo {file} do S3: {e}")
            raise

    return tuple(files_content)



@task(log_prints=True)
def data_ingest(file_parquet):
    """
    Prepara os dados de entrada para treinamento ou inferência.

    Converte um arquivo Parquet em um DataFrame, verifica colunas obrigatórias, calcula a duração 
    das viagens, filtra valores inválidos e seleciona as colunas necessárias. Retorna os atributos 
    preditores (X) e o alvo (y).

    Parameters:
    -----------
    file_parquet : str or bytes
        Caminho ou fluxo de bytes do arquivo Parquet.

    Returns:
    --------
    tuple
        X : pd.DataFrame - Atributos preditores.
        y : np.ndarray - Duração das viagens em minutos.
    """

    # Converte os arquivos Parquet baixados em DataFrames pandas
    X = pq.read_table(BytesIO(file_parquet)).to_pandas()

    # Garantir que X seja um DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Os dados de entrada devem ser fornecidos como DataFrame.")

    # Verificar features necessárias
    required_features = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'trip_distance', 'DOLocationID', 'PULocationID']
    missing_features = [col for col in required_features if col not in X.columns]
    if missing_features:
        raise ValueError(f"Faltam as colunas: {', '.join(missing_features)}")

    # Processamento de datas
    X['lpep_pickup_datetime'] = pd.to_datetime(X['lpep_pickup_datetime'], errors='coerce')
    X['lpep_dropoff_datetime'] = pd.to_datetime(X['lpep_dropoff_datetime'], errors='coerce')
    X['duration'] = (X['lpep_dropoff_datetime'] - X['lpep_pickup_datetime']).dt.total_seconds() / 60

    # Filtrar valores inválidos
    X = X[(X.duration >= 1) & (X.duration <= 60)]

    # Criar identificador único
    X['rideID'] = X['lpep_pickup_datetime'].dt.strftime("%Y/%m_") + X.index.astype(str)

    # Criando a feature PU_DO_LocationID
    X['PU_DO_LocationID'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)

    # Selecionar os atributos eleitos
    #final_features = ['rideID', 'PU_DO_LocationID', 'trip_distance', 'duration']

    # Atributos preditores
    #X = X[final_features]

    # Atributo a ser predito
    y = X.pop('duration').values
    time.sleep(3)
    return X, y
################################################################################
#                                Data Ingestion                                #
################################################################################
#-------------------------------------End--------------------------------------#


#---------------------------------- Begin -------------------------------------#
################################################################################
#                         Data Preprocessing Pipeline                          #
################################################################################

class TransformerWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper para integração com o MLflow.

    Esta classe encapsula um 'ColumnTransformer' da função dataset_tranformations em 
    compatibilidade com o formato PyFunc do MLflow, permitindo que seja registrado,
    armazenado e carregado em tempo de execução como um artefato do mlflow.
    
    A classe é ideal para aplicações cloud-native, pois independe de bibliotecas
    de serialização legadas como pickle e cloudpickle, facilitando a portabilidade. 
    """
    def __init__(self, transformer: ColumnTransformer):
        """
        Inicializa o wrapper do transformer para log_artifact com mlflow.

        Parameters:
        -----------
        transformer : ColumnTransformer
            O transformador ajustado para pré-processar os dados.
        """
        self.transformer = transformer

    def predict(self, context, model_input):
        """
        Aplica o pré-processamento ao conjunto de dados de entrada.

        Parameters:
        -----------
        context : dict
            Contexto de execução fornecido pelo MLflow (não utilizado diretamente no pipeline).

        model_input : Matrix-like
            Os dados de entrada que precisam ser transformados. Devem estar no formato 'pd.DataFrame'.


        Returns:
        --------
        np.ndarray
            Os dados transformados pelo pipeline.

                    
        """
        return self.transformer.transform(model_input)



# Tarefa para ajustar e executar o pipeline completo de preprocessamento
@task(log_prints=True)
def data_cleasing(data,train_mode=True, preprocessor=None):
    """
    Pipeline de preprocessamento de dados para o nyc-trip-data.

    Parameters:
    -----------
    data : pd.DataFrame
        Dados de entrada já carregados.

    exp_name : str
        Nome do experimento para registro no MLflow.

    train_mode : bool, default=True
        Se True, realiza o ajuste do pipeline e retorna X.
        Se False, utiliza o pré-processador fornecido para transformar novos dados.

    preprocessor : DataPreprocessorWrapper, default=None
        Wrapper do pipeline ajustado para uso em modo de teste.

    Returns:
    --------
    X_processed : np.ndarray
        Dados transformados e prontos para inferência.

    preprocessor_wrapper : preprocessor_wrapper
        Objeto do pipeline completo compatível com o MLflow.

    run_id : str
        ID do experimento rastreado pelo MLflow.
    """
    logger = get_run_logger()

    if data is None:
        raise ValueError("Os dados devem ser fornecidos como DataFrame para o pipeline.")

    

    # Definir colunas numéricas e categóricas
    num_features = ['trip_distance']
    nominal_features = ['PU_DO_LocationID']

    if train_mode:
        logger.info(f"Aplicando a etapa de preprocessamento - Dados de treinamento...")

        # Prefect*
        create_link_artifact(
            key="train",  # Chave indicando artefato de treino
            link="",  # Link fictício para dados de treino
            description="Preprocessor nos dados de treino - metodo fit_transform()"
        )        

        # Pipeline para atributos numéricos
        num_pipeline = Pipeline([
            #('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', RobustScaler())
        ])

        # Pipeline para atributos categóricos
        nominal_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combinação de pré-processadores para colunas numéricas e categóricas
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
            ('cat', nominal_pipeline, nominal_features)
        ])

        # Ajustar pipeline e transformar dados
        X_processed = preprocessor.fit_transform(data)

        # Criar wrapper do pipeline completo para registro no MLflow
        mlflow_preprocessor =  TransformerWrapper(preprocessor)



        # Retornar os dados processados e o mlflow_preprocessor ajustado
        return X_processed, mlflow_preprocessor

    else:
        # Inferência
        if not preprocessor:
            raise ValueError("Preprocessor ajustado deve ser fornecido no modo de inferência.")
        logger.info(f"Aplicando a etapa de preprocessamento - Dados de teste...")
        create_link_artifact(
            key="test",  # Chave indicando artefato de treino
            link="",  # Link fictício para dados de treino
            description="Preprocessor nos dados de teste - metodo transform()"
        )

        # Transform nos dados usando o preprocessor_wrapper via metodo predict
        X_processed = preprocessor.predict(context=None, model_input=data)

        return X_processed
################################################################################
#                         Data Preprocessing Pipeline                          #
################################################################################
#----------------------------------- End --------------------------------------#


#---------------------------------- Begin -------------------------------------#
################################################################################
#                           Model Train Pipeline                               #
################################################################################


@task(log_prints=True)
def train_model(model, hyperparams_grid , random_search_params, X_train, y_train):
    """
    Função integrada ao pipTrain para fitar cada modelo usando RandomizedSearchCV.

    Args:
        model: Modelo ou pipeline a ser treinado.
        hyperparameters (list): Grade de hiperparâmetros para otimização.
        random_search_params: Parâmetros extras para o RandomizedSearchCV (como scoring, n_iter, cv, etc).
        X_train (array-like): Dados de treino.
        y_train (array-like): Alvo de treino.

    Returns:
        tuple: Melhor modelo treinado com o melhor conjunto de hiperparâmetros obtidos pelo RandomizedSearchCV.
    """
    warnings.filterwarnings("ignore", message=".*The total space of parameters.*")
    
    hp_search = RandomizedSearchCV(
        model,
        param_distributions=hyperparams_grid,
        random_state=42,
        **random_search_params
    )
    hp_search.fit(X_train, y_train)
    return hp_search.best_estimator_, hp_search.best_params_


@task(log_prints=True)
def objective(
    model_name,
    best_model,
    best_params,
    X_train,
    X_test,
    y_train,
    y_test
):  
    """
    Esta função recebe cada modelo fitado na função train_model com seus 
    respectivos melhores hiperparâmetros, e realiza as previsões fornecidos
    e grava os resultados no MLflow.

    Args:
        model_name (str): Nome do modelo que está sendo treinado (por exemplo, 'LinearRegression').
        best_model: Modelo treinado com os melhores hiperparâmetros.
        preprocessor (str): Nome do pré-processador utilizado (por exemplo, 'StandardScaler').
        best_params (dict): Dicionário contendo os melhores hiperparâmetros encontrados para o modelo.
        X_train (array-like): Dados de treino (features) usados para treinar o modelo.
        X_test (array-like): Dados de teste (features) usados para avaliar o modelo.
        y_train (array-like): Valores alvo (target) de treino para o modelo.
        y_test (array-like): Valores alvo (target) de teste para avaliação do modelo.
        exp_name (str, opcional): Nome do experimento no MLflow, usado para organizar e trackear as execuções.

    Returns:
        None: A função realiza o treinamento, avaliação e log de parâmetros, métricas e artefatos no MLflow.

    """
    logger = get_run_logger()

    # Filtrar e omitir apenas o warning de signature do mlflow
    logging.getLogger("mlflow.models.model").addFilter(
        lambda record: "Model logged without a signature and input example" not in record.getMessage()
    )


    run = mlflow.active_run()  # Obtém a run ativa

    # Nome aleatório gerado pelo MLflow da run atual (sem criar nova)
    generated_name = run.data.tags.get("mlflow.runName", run.info.run_id)
    
    # Criar um novo nome concatenando com o nome do modelo
    custom_run_name = f"{model_name}_{generated_name}"
    
    # Definir o novo nome na run ativa
    mlflow.set_tag("mlflow.runName", custom_run_name)
    mlflow.set_tag("author", "Wanderson")

    # salvar o melhor modelo treinado no mlflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="models",
        input_example=X_train[:1],
        signature=infer_signature(X_train, best_model.predict(X_train))
    )


    # Predições
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Métricas
    metrics_train = {
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "r2_train": r2_score(y_train, y_pred_train)
    }
    metrics_test = {
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "r2_test": r2_score(y_test, y_pred_test)
    }

    
    # Log das métricas no MLflow
    for key, value in {**metrics_train, **metrics_test}.items():
        mlflow.log_metric(key, value)
    # Log dos melhores parâmetros
    for key, value in best_params.items():
        mlflow.log_param(key, value)

    mlflow.log_param("model_name", model_name)
    mlflow.set_tag("model_type", model_name)  # Ajuda na organização

    logger.info(f"Modelo {model_name} avaliado e logado.")


    # Criar chave sanitizada para o table artifact no prefect
    sanitized_key = f"{model_name}-metrics-table".lower().replace("_", "-")

    # Criar a tabela de métricas como artifact para o prefect
    table_data = [
        {"Dataset": "Train", "RMSE": metrics_train["rmse_train"], "R2": metrics_train["r2_train"]},
        {"Dataset": "Test", "RMSE": metrics_test["rmse_test"], "R2": metrics_test["r2_test"]}
    ]
    table_markdown = create_table_artifact(
        key=sanitized_key,
        table=table_data,
        description=f"Métricas de desempenho do modelo {model_name}."
    )
    logger.info(f"Tabela de métricas criada:\n{table_markdown}")



@task(log_prints=True)
def pipTrain(exp_name,
              s3bucket_name,
              s3bucket_mlflow_artifact_folder,
              models, 
              preprocessor, 
              hyperparameters, 
              random_search_params, 
              X_train, X_test, y_train, y_test):
    """
    Executa o pipeline de treinamento com runs aninhadas no MLflow.
    Cada n modelo será treinado dentro de um run filho (child).

    Args:
        exp_name (str): Nome do experimento.
        run_id (str): Run_id do pré-processador.
        models (list): Lista de pipelines de modelos a serem treinados.
        hyperparameters (list): Hiperparâmetros para cada modelo.
        random_search_params: Parâmetros extras para o RandomizedSearchCV (como scoring, n_iter, cv, etc).
        X_train (pd.DataFrame): Dados de treino.
        X_test (pd.DataFrame): Dados de teste.
        y_train (pd.Series): Labels de treino.
        y_test (pd.Series): Labels de teste.
    """
    logger = get_run_logger()

    aws_credentials_block = AwsCredentials.load("aws-s3-creds")
    os.environ["AWS_ACCESS_KEY_ID"] = aws_credentials_block.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_credentials_block.aws_secret_access_key.get_secret_value()


    parent_run_id = configure_mlflow(exp_name, s3bucket_name, s3bucket_mlflow_artifact_folder)  # Criar uma run pai

    # Criar run filha para o pré-processador dentro da run pai
    with mlflow.start_run(run_id=parent_run_id, nested=True) as parent_run:
        with mlflow.start_run(nested=True):  # Run filha do preprocessor
            mlflow.set_tag("author", "Wanderson")
            mlflow.set_tag("run_type", "preprocessor")
            # Definir um nome customizado para a run ativa
            run = mlflow.active_run() 
            generated_name = run.data.tags.get("mlflow.runName", run.info.run_id)
            custom_run_name = f"Preprocessor_{generated_name}"            
            mlflow.set_tag("mlflow.runName", custom_run_name)
            # salva o preprocessador no mlflow
            mlflow.pyfunc.log_model(
                python_model=preprocessor,
                artifact_path="preprocessor",
                conda_env=mlflow.pyfunc.get_default_conda_env(),
            )
            print(f"✅ Pré-processador registrado no MLflow como run filho (Run ID: {mlflow.active_run().info.run_id})")

        # Este loop permite que a função train_model receba cada 
        # modelo com sua respectiva grade de hiperparametros
        for i, model in enumerate(models): #  
            model_name = type(model.steps[-1][1]).__name__
            print(f"🏋️‍♂️ Treinando modelo: {model_name}")
            
            best_model, best_hp = train_model(
                model = model, 
                hyperparams_grid = hyperparameters[i], 
                random_search_params = random_search_params,
                X_train = X_train, 
                y_train = y_train)
            
            # Criar uma run filha para cada modelo dentro da run pai
            with mlflow.start_run(nested=True):  
                objective(
                    model_name=model_name,
                    best_model=best_model,
                    best_params=best_hp,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )    
        logger.info("✅ Pipeline de treinamento concluído!")




################################################################################
#                           Model Train Pipeline                               #
################################################################################
#----------------------------------- End --------------------------------------#

    

################################################################################
#                           Apply Full Pipeline                                #
################################################################################
# Fluxo principal para execução de todas as etapas acima:                      # 
#                                                                              #
# aws_fetch_data                                                             # 
#     |--> ingest_data                                                         # 
#               |--> data_cleasing --{TranformerWrapper}               # 
#                        |--> pipTrain --{train_model --> objective}    #
#                                                                              #
################################################################################
#-----------------------------------Begin--------------------------------------#


@flow(log_prints=True)
async def main_flow(exp_name : str = None,
                    taxi_type : str = None,
                    date: str = None, 
                    s3bucket_name: str = None, 
                    s3bucket_dataset_folder: str = None,
                    s3bucket_mlflow_artifact_folder : str = None):
    """
    Fluxo principal que treina modelos com base em dados dinâmicos por data.
    """
    logger = get_run_logger()

    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    train_raw_file, test_raw_file = await aws_fetch_data(
        taxi_type, 
        date, 
        s3bucket_name, 
        s3bucket_dataset_folder
    )
    logger.info(f"🎲 Dados de treino e teste obtidos como DataFrame.")

    # Limpeza e carregamento dos conjuntos de dados
    train_df, y_train = data_ingest(train_raw_file)
    test_df, y_test = data_ingest(test_raw_file)

    # Definir colunas numéricas e categóricas
    numerical_features=['trip_distance']
    categorical_features=['PU_DO_LocationID']

    # Treinamento
    X_train_processed, mlflow_preprocessor = data_cleasing(
        data=train_df[numerical_features + categorical_features], 
        train_mode=True
    )

    logger.info(f"Tamanho de X_train: {X_train_processed.shape}")

    # Inferência (Teste): reutilizar o preprocessor ajustado no pipeline
    X_test_processed = data_cleasing(
        data=test_df[numerical_features + categorical_features],
        train_mode=False,
        preprocessor=mlflow_preprocessor
    )

    logger.info(f"Tamanho de X_test: {X_test_processed.shape}")

    # Criar a lista de modelos 
    models = [
        Pipeline([('LinearRegression', LinearRegression())]),
        Pipeline([('Ridge', Ridge())]),
        Pipeline([('Lasso', Lasso())]),
        Pipeline([('xgb-regressor', XGBRegressor())])
    ]

    # Hiperparâmetros para os modelos
    hyperparameters = [
        {},  # Linear Regression não precisa de hiperparâmetros para esta abordagem
        {'Ridge__alpha': [0.01, 0.1, 1, 5, 10]},
        {'Lasso__alpha': [0.01, 0.1, 1, 5, 10]},
        {
         'xgb-regressor__n_estimators': [200, 300, 400],
         'xgb-regressor__learning_rate': [0.05, 0.1, 0.15],
         'xgb-regressor__max_depth': [6, 8, 10],
         'xgb-regressor__min_child_weight': [3, 5, 7],
         'xgb-regressor__gamma': [0.3, 0.5, 0.7],
         'xgb-regressor__subsample': [0.6, 0.8, 1.0],
         'xgb-regressor__colsample_bytree': [0.4, 0.5, 0.6],
         'xgb-regressor__reg_alpha': [0, 0.5, 1],
         'xgb-regressor__reg_lambda': [0.5, 1, 1.5],
         'xgb-regressor__seed': [42]
        }
    ]


    # Parâmetros para o RandomizedSearchCV
    random_search_params={
         'n_iter': 10,
         'scoring': 'r2',  
         'cv': 3,
         'verbose':1,
         'n_jobs':-1,          
         # outros parâmetros, se necessário
    }


    # Executar o fluxo de treinamento
    pipTrain(
        exp_name=exp_name,
        s3bucket_name = s3bucket_name,
        s3bucket_mlflow_artifact_folder = s3bucket_mlflow_artifact_folder,
        models=models,
        preprocessor = mlflow_preprocessor,
        hyperparameters=hyperparameters,
        random_search_params = random_search_params,
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test
    )


    print(f"Fluxo principal concluído para a data: {date}")

'''
if __name__ == "__main__":
    import asyncio

    asyncio.run(main_flow(
        exp_name='nyc-green-taxi-2024-01',
        taxi_type='green',
        date='2024-01-01',
        s3bucket_name='mlflow-cloud-artifacts',
        s3bucket_dataset_folder='nyc-trip-data',
        s3bucket_mlflow_artifact_folder='nyc-mlflow-artifacts'
    ))
'''

################################################################################
#                           Apply Full Pipeline                                #
################################################################################
#----------------------------------- End --------------------------------------#

# Instructions for train.py:
#------------------------------localhost--------------------------------------
# 1 - (bash terminal): conda activate mlops-env
# 2 - (bash terminal): prefect server start
# 3 - Paste on web browser: http://127.0.0.1:4200/ 
#------------------------------------------------------------------------
#-------------------------aws virtual Machine-------------------------------
# 4 - (bash terminal): ssh -i ~/.ssh/vm-mlops_key.pem awsuser@191.235.81.94
# 5 - (bash terminal): conda activate nyc-env
# 6 - (bash terminal): nohup ./start_mlflow.sh > mlflow.log 2>&1 & 
# 7 - Paste on web browser: http://191.235.81.94:5000/
#-----------------------------------------------------------------------------
#------------------------------localhost--------------------------------------
# 8 - prefect worker start --pool "vm-mlops"
# 9 - (bash terminal - mlopsenv): python3.12 train.py 
