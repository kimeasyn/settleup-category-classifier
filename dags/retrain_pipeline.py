from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.amazon.aws.operators.ec2 import EC2StartInstanceOperator, EC2StopInstanceOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, time

INSTANCE_ID = os.environ.get("EC2_INSTANCE_ID")
REGION = os.environ.get("EC2_REGION", "ap-northeast-2")

EC2_BASE = "cd /home/ubuntu/app/settleup-category-classifier && source /home/ubuntu/app/.env && source venv/bin/activate"
EC2_EXTRACT = f"{EC2_BASE} && python retrain/extract_data.py"
EC2_TRAIN = f"{EC2_BASE} && python retrain/train_model.py"
EC2_CONVERT = f"{EC2_BASE} && python retrain/convert_onnx.py"


def wait_for_ec2():
    """EC2 부팅 + Tailscale 연결 대기"""
    time.sleep(60)

with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=6),
) as dag:

    start_ec2 = EC2StartInstanceOperator(
        task_id="start_ec2",
        instance_id=INSTANCE_ID,
        region_name=REGION,
        aws_conn_id="aws_default",
    )

    wait = PythonOperator(
        task_id="wait_for_boot",
        python_callable=wait_for_ec2,
    )

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

    stop_ec2 = EC2StopInstanceOperator(
        task_id="stop_ec2",
        instance_id=INSTANCE_ID,
        region_name=REGION,
        aws_conn_id="aws_default",
        trigger_rule="all_done",
    )

    start_ec2 >> wait >> extract >> train >> convert >> stop_ec2

    extract >> train >> convert