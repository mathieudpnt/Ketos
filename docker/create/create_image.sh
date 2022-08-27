# before running this script, make sure to login to 
# docker with `docker login --username meridiancfi`

# copy package and requirements
cp -r ../../ketos/ .
cp ../../setup.py .
cp ../../requirements.txt .

# build image
docker build --tag=ketos_v2.6.1 .

# tag image
docker tag ketos_v2.6.1 meridiancfi/ketos:v2.6.1

# push image to repository
docker push meridiancfi/ketos:v2.6.1

# clean
rm -rf ketos
rm -rf setup.py
rm -rf requirements.txt
