import json
import heapq
from datetime import datetime

num = 0

with open("../mastodon-144Gb.ndjson", "r") as f:
    for line in f:
        num += 1

print(num)
