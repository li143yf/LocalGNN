import pickle
try:
    with open("/lf/wjz/temp/cache", "rb") as f:
        cache = pickle.load(f)
except:
    cache = None
else:
    cache = None

print(cache)