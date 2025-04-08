import json
import heapq
from datetime import datetime
from mpi4py import MPI
import os
import time as t

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

filename = "../mastodon-106k.ndjson"

timer = t.time()

# Get total file size
file_size = os.path.getsize(filename)
chunk_size = file_size // size
start = rank * chunk_size
end = file_size if rank == size - 1 else (rank + 1) * chunk_size
# FIX: SKIPPING LINES

user_hp = []
user_map = {}

time_hp = []
time_map = {}

# print(f"Rank {rank} reading bytes from {start} to {end - 1}")

# OPTIMIZE: SEND LISTS NOT MAP
with open(filename, 'r', encoding='utf-8') as f:
    # Move to start offset
    f.seek(start)

    # Skip partial line if not rank 0
    if start != 0:
        f.readline()

    current_position = f.tell()

    while current_position < end:
        line = f.readline()
        
        if not line:
            break  # EOF
        # Process the line
        #print(f"Rank {rank} read: {line.strip()} END%^&**(*^%%$)")
        current_position = f.tell()

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        username = item.get("doc", {}).get("account", {}).get("username")
        user_id = item.get("doc", {}).get("account", {}).get("id")
        sentiment = item.get("doc", {}).get("sentiment")
        createdAt = item.get("doc", {}).get("createdAt")

        if user_id is None or username is None or sentiment is None or createdAt is None:
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
        
all_user_map = comm.gather(user_map, root = 0)
all_time_map = comm.gather(time_map, root = 0)

# print("number of lines: " + str(num))

if rank == 0:

    total_user_map = {}
    total_time_map = {}

    for process_user_map in all_user_map:
        for user, sentiment in process_user_map.items():
            total_user_map[user] = total_user_map.get(user, 0) + sentiment

    for process_time_map in all_time_map:
        for time_key, sentiment in process_time_map.items():
            total_time_map[time_key] = total_time_map.get(time_key, 0) + sentiment

    top_5_users = heapq.nlargest(5, total_user_map.items(), key=lambda x: x[1])
    bottom_5_users = heapq.nsmallest(5, total_user_map.items(), key=lambda x: x[1])
    top_5_times = heapq.nlargest(5, total_time_map.items(), key=lambda x: x[1])
    bottom_5_times = heapq.nsmallest(5, total_time_map.items(), key=lambda x: x[1])

    run_time = t.time() - timer

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

