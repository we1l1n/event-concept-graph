### OpenKE
```
docker run -ti -v e:/docker/openke:/work --name openke -p 8888:8888 tensorflow/tensorflow
git clone https://github.com/thunlp/OpenKE.git
cd OpenKE
bash make.sh
g++ ./base/Base.cpp -fPIC -shared -o ./release/Base.so -pthread -O3 -march=native

./task/train2id.txt,entity2id.txt,relation2id.txt
python train_TransX.py
result in ./task/E(DHR)/E(DHR).vec.json

import json
E = json.load(open(E.vec.json))
E['ent_embeddings']
E['rel_embeddings']
```