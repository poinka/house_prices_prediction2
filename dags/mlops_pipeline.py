from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dag = DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Automated MLOps Pipeline',
    schedule='*/5 * * * *',
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

process_data = BashOperator(
    task_id='data_engineering',
    bash_command=f'python "{os.path.join(project_root, "code/datasets/process_data.py").replace(" ", "\\ ")}"',
    dag=dag,
)

train_model = BashOperator(
    task_id='model_engineering',
    bash_command=f'python "{os.path.join(project_root, "code/models/train.py").replace(" ", "\\ ")}"',
    dag=dag,
)

deploy = BashOperator(
    task_id='deployment',
    bash_command=f'cd "{os.path.join(project_root, "code/deployment").replace(" ", "\\ ")}" && docker-compose up -d --build',
    dag=dag,
)

process_data >> train_model >> deploy