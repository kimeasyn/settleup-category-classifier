from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime, timedelta

EC2_BASE = "cd /home/ubuntu/app/settleup-category-classifier && source /home/ubuntu/app/.env && source venv/bin/activate"
EC2_EXTRACT = f"{EC2_BASE} && python retrain/extract_data.py"
EC2_TRAIN = f"{EC2_BASE} && python retrain/train_model.py"
EC2_CONVERT = f"{EC2_BASE} && python retrain/convert_onnx.py"

with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=3),
) as dag:

    extract = SSHOperator(
        task_id="extract_data",
        ssh_conn_id="ec2_retrain",
        command=EC2_EXTRACT,
        execution_timeout=timedelta(hours=1),
        cmd_timeout=None,
    )

    train = SSHOperator(
        task_id="train_model",
        ssh_conn_id="ec2_retrain",
        command=EC2_TRAIN,
        execution_timeout=timedelta(hours=2),
        cmd_timeout=None,
    )

    convert = SSHOperator(
        task_id="convert_onnx",
        ssh_conn_id="ec2_retrain",
        command=EC2_CONVERT,
        execution_timeout=timedelta(hours=1),
        cmd_timeout=None,
    )

    extract >> train >> convert