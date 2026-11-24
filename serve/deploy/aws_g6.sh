#!/usr/bin/env bash
# Deploy Querycraft on ONE AWS GPU instance (g6.xlarge: 1x NVIDIA L4 24 GB). NOT RUN by this project ($0 budget).
# Costs money from the moment the instance starts. Read serve/deploy/DEPLOY.md first. Requires: aws CLI configured,
# a key pair, and a security group that allows 22 and 8000 from YOUR IP only.
set -euo pipefail
: "${KEY_NAME:?set KEY_NAME}" "${SG_ID:?set SG_ID (security group)}" "${REGION:=us-east-1}"
AMI=$(aws ssm get-parameter --region "$REGION" --query Parameter.Value --output text \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id)
ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type g6.xlarge --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=querycraft}]' --query 'Instances[0].InstanceId' --output text)
echo "instance $ID starting (billing has started)"
aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
# ship the repo with the trained artifacts (adapter, retriever, abstain model) and the databases to serve
rsync -az --exclude .venv --exclude data/raw/train --exclude 'runs/*/eval' -e ssh ./ "ubuntu@$IP:querycraft/"
ssh "ubuntu@$IP" 'cd querycraft && docker compose -f serve/docker-compose.yml up -d --build'
echo "API: http://$IP:8000   (stop billing: aws ec2 terminate-instances --instance-ids $ID --region $REGION)"
