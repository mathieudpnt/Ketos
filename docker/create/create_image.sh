
# copy package and requirements
cp -r ../../ketos/ .
cp ../../setup.py .
cp ../../requirements.txt .
cp ../../../meridian-rtd-theme/dist/sphinx_mer_rtd_theme-0.4.3.dev0.tar.gz .

# build image
docker build --tag=ketos_v2.4.0 .

# tag image
docker tag ketos_v2.4.0 meridiancfi/ketos:v2.4.0

# push image to repository
docker push meridiancfi/ketos:v2.4.0

# clean
rm -rf ketos
rm -rf setup.py
rm -rf requirements.txt
rm -rf sphinx_mer_rtd_theme-0.4.3.dev0.tar.gz
