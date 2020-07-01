wget -nc https://motchallenge.net/data/MOT17Det.zip -P data/
wget -nc https://motchallenge.net/data/MOT16Labels.zip -P data/
wget -nc https://motchallenge.net/data/2DMOT2015.zip -P data/
wget -nc https://motchallenge.net/data/MOT16-det-dpm-raw.zip -P data/
wget -nc https://motchallenge.net/data/MOT17Labels.zip -P data/

unzip -d 'data/MOT17Det' 'data/MOT17Det.zip'
unzip -d 'data/MOT16Labels' 'data/MOT16Labels.zip'
unzip -d 'data/2DMOT2015' 'data/2DMOT2015.zip'
unzip -d 'data/MOT16-det-dpm-raw' 'data/MOT16-det-dpm-raw.zip'
unzip -d 'data/MOT17Labels' 'data/MOT17Labels.zip'
