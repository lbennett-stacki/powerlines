KEY_C="lbennett@fastmail.com"
HOST="lb@192.168.1.124"
KEY_NAME="wsl2_id_rsa"
PORT=2222

ssh-keygen -t rsa -b 4096 -C $KEY_C -f $KEY_NAME
ssh-copy-id -p $PORT -i $KEY_NAME $HOST
ssh -p $PORT -i $KEY_NAME $HOST 
