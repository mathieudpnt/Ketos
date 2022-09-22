#!/bin/bash

ssh ubuntu@docsvm << EOF
sudo mkdir /var/www/html/ketos2
sudo chown -h -R ubuntu:ubuntu /var/www/html/ketos2
sudo chown -h -R ubuntu:ubuntu /var/www/html/ketos
EOF

# copy html folder
scp -r build/html/* ubuntu@docsvm:/var/www/html/ketos2

# replace old folder on server, and set permissions
ssh ubuntu@docsvm << EOF
cd /var/www/html/
sudo rm -rf ketos
sudo mv ketos2 ketos
sudo chown -h -R www-data:www-data ketos
sudo chmod -R g+w ketos
EOF
