#!/bin/bash

# prompt user for username to MERIDIAN's docs server
echo 'Enter username'
read user

# copy html folder
scp -P 23022 -r build/html $user@206.12.88.81:/var/www/html/ketos2

# replace old folder on server, and set permissions
ssh -p 23022 $user@206.12.88.81 << EOF
cd /var/www/html/
rm -rf ketos
mv ketos2 ketos
sudo chown -h -R www-data:www-data ketos
sudo chmod -R g+w ketos
EOF

