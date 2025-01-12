HOST="$1"
USER=lb
KEY_NAME=wsl2_id_rsa
PORT=2222

if [[ -z "$HOST" ]]; then
  echo "Usage: join.sh <host-ip>"
  exit 1
fi

ssh -p $PORT -i $KEY_NAME "$USER@$HOST"
