HOST="$1"
USER=lb
KEY_NAME=wsl2_id_rsa
PORT=2222

if [[ -z "$HOST" ]]; then
  echo "Usage: copy-data.sh <host-ip>"
  exit 1
fi

ssh -p $PORT -i $KEY_NAME "$USER@$HOST" "zsh -c 'rm -rf /home/lb/workspaces/personal/powerlines/data && rm -rf /home/lb/workspaces/personal/powerlines/data.zip'"
zip -r ../data.zip ../data
scp -r -P $PORT -i $KEY_NAME ../data.zip "$USER@$HOST:/home/lb/workspaces/personal/powerlines/data.zip"
ssh -p $PORT -i $KEY_NAME "$USER@$HOST" "zsh -c 'unzip /home/lb/workspaces/personal/powerlines/data.zip -d /home/lb/workspaces/personal/powerlines/data'"
