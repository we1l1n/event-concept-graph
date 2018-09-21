```
sudo docker run -v /autophrase/data:/autophrase/data -v /autophrase/results:/autophrase/results -it \
    remenberl/autophrase

edit autophrase.sh:
>> RAW_TRAIN = news_event.txt[one sentence per line]

./autophrase.sh
```