import json
import time
s = [0 for i in range(2**20)]
for i in range(2**20):
    string = bin(i)
    if '11111' in string:
        s[i] = 1

t1 = time.time()
for i in range(2**20):
    x = '11111' in bin(i)
t1 = time.time() - t1


t2 = time.time()
for i in range(2**20):
    x = s[int(bin(i), 2)]
t2 = time.time() - t2

print(t1, t2)

# with open('detect 5.json', 'w+') as f:
#     json.dump(s, f)
