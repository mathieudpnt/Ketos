
# copy package and requirements
cp -r ../../ketos/ .
cp ../../setup.py .
cp ../../requirements.txt .

# build image
docker build --tag=ketos_test1 .

# tag image
docker tag ketos_test1 oliskir/ketos:test1

# push image to repository
docker push oliskir/ketos:test1

# clean
rm -rf ketos
rm -rf setup.py
rm -rf requirements.txt
