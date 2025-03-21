import json
import heapq
from datetime import datetime
from mpi4py import MPI

user_hp = []
user_map = {}

time_hp = []
time_map = {}

num = 0

with open("../mastodon-106k.ndjson", "r") as f:
    for line in f:
        num += 1
        item = json.loads(line)
        username = item.get("doc", {}).get("account", {}).get("username")
        sentiment = item.get("doc", {}).get("sentiment")
        createdAt = item.get("doc", {}).get("createdAt")
        if username is None and sentiment is None and createdAt is None:
            continue

        dt = datetime.fromisoformat(createdAt.replace("Z", "+00:00"))
        time = f"{dt.date().isoformat()} {dt.hour:02d}"

        if username in user_map:
            user_map[username] += sentiment
        else:
            user_map[username] = sentiment

        if time in time_map:
            time_map[time] += sentiment
        else:
            time_map[time] = sentiment
        
top_5_users = heapq.nlargest(5, user_map.items(), key=lambda x: x[1])
bottom_5_users = heapq.nsmallest(5, user_map.items(), key=lambda x: x[1])
top_5_times = heapq.nlargest(5, time_map.items(), key=lambda x: x[1])
bottom_5_time = heapq.nsmallest(5, time_map.items(), key=lambda x: x[1])

print("5 happiest persons are", top_5_users)
print("5 saddest persons are", bottom_5_users)
print("5 happiest hour are", top_5_times)
print("5 saddest hour are", bottom_5_time)
print(num)






