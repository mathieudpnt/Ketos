#!/bin/bash

# prompt user for username to MERIDIAN's docs server
echo 'Enter username'
read user

ssh -p 22 $user@206.12.88.81 << EOF
sudo mkdir /var/www/html/ketos2
sudo chown -h -R ubuntu:ubuntu /var/www/html/ketos2
sudo chown -h -R ubuntu:ubuntu /var/www/html/ketos
EOF

# copy html folder
scp -P 22 -r build/html/* $user@206.12.88.81:/var/www/html/ketos2

# replace old folder on server, and set permissions
ssh -p 22 $user@206.12.88.81 << EOF
cd /var/www/html/
sudo rm -rf ketos
sudo mv ketos2 ketos
sudo chown -h -R www-data:www-data ketos
sudo chmod -R g+w ketos
EOF

