# before running this script, make sure to login to 
# docker with `docker login --username meridiancfi`

# copy package and requirements
cp -r ../../ketos/ .
cp ../../setup.py .
cp ../../requirements.txt .

# build image
docker build --tag=ketos_v2.6.2 .

# tag image
docker tag ketos_v2.6.2 meridiancfi/ketos:v2.6.2

# push image to repository
docker push meridiancfi/ketos:v2.6.2

# clean
rm -rf ketos
rm -rf setup.py
rm -rf requirements.txt
