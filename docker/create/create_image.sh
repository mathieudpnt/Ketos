
# copy package and requirements
cp -r ../../ketos/ .
cp ../../setup.py .
cp ../../requirements.txt .
cp ../../../sphinx_mer_rtd_theme-0.4.3.dev0.tar.gz .

# build image
docker build --tag=ketos_v2.0.1 .

# tag image
docker tag ketos_v2.0.1 meridiancfi/ketos:v2.0.1

# push image to repository
docker push meridiancfi/ketos:v2.0.1

# clean
rm -rf ketos
rm -rf setup.py
rm -rf requirements.txt
rm -rf sphinx_mer_rtd_theme-0.4.3.dev0.tar.gz
