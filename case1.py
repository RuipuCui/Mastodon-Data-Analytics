import json
import heapq
from datetime import datetime
from mpi4py import MPI
import os
import time as t

timer = t.time()

user_hp = []
user_map = {}

time_hp = []
time_map = {}

with open("../mastodon-144g.ndjson", "r") as f:
    for line in f:
        item = json.loads(line)
        username = item.get("doc", {}).get("account", {}).get("username")
        user_id = item.get("doc", {}).get("account", {}).get("id")
        sentiment = item.get("doc", {}).get("sentiment")
        createdAt = item.get("doc", {}).get("createdAt")

        if username is None or sentiment is None or createdAt is None:
            continue

        try:
            dt = datetime.fromisoformat(createdAt.replace("Z", "+00:00"))
        except ValueError:
            continue
        time = f"{dt.date().isoformat()} {dt.hour:02d}"

        user_info = str(user_id) + '$' +username

        if user_info in user_map:
            user_map[user_info] += sentiment
        else:
            user_map[user_info] = sentiment

        if time in time_map:
            time_map[time] += sentiment
        else:
            time_map[time] = sentiment
        
top_5_users = heapq.nlargest(5, user_map.items(), key=lambda x: x[1])
bottom_5_users = heapq.nsmallest(5, user_map.items(), key=lambda x: x[1])
top_5_times = heapq.nlargest(5, time_map.items(), key=lambda x: x[1])
bottom_5_times = heapq.nsmallest(5, time_map.items(), key=lambda x: x[1])

run_time = t.time() - timer

print("case 1: 1 nodes, 1 cores")

print(f"\nProgram ran for: {run_time:.2f} seconds\n")

print("***** TOP 5 USERS *****")
for user_info, sentiment in top_5_users:
    uid, uname = user_info.split('$', 1)
    print(f"User ID: {uid:<22} Username: {uname:<20} Total Sentiment: {sentiment:.4f}")

print("\n***** TOP 5 TIMES *****")
for time_key, sentiment in top_5_times:
    print(f"Time: {time_key:<16} Total Sentiment: {sentiment:.4f}")

print("\n***** BOTTOM 5 USERS *****")
for user_info, sentiment in bottom_5_users:
    uid, uname = user_info.split('$', 1)
    print(f"User ID: {uid:<22} Username: {uname:<20} Total Sentiment: {sentiment:.4f}")

print("\n***** BOTTOM 5 TIMES *****")
for time_key, sentiment in bottom_5_times:
    print(f"Time: {time_key:<16} Total Sentiment: {sentiment:.4f}")






