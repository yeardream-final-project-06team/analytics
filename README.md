# analytics

```bash
git clone https://github.com/yeardream-final-project-06team/analytics.git

cd analyatics

docker build -t analytics ./

# if use gpu
docker run -it -v ./:/analytics -u 1000:1000 --gpus all analytics python3 /analytics/dl_train.py

# if use cpu
docker run -it -v ./:/analytics -u 1000:1000 analytics python3 /analytics/dl_train.py
```