import json
import heapq
from datetime import datetime
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

filename = "../mastodon-16m.ndjson"

# Get total file size
file_size = os.path.getsize(filename)
chunk_size = file_size // size
start = rank * chunk_size
end = file_size if rank == size - 1 else (rank + 1) * chunk_size

num = 0
user_hp = []
user_map = {}

time_hp = []
time_map = {}

print(f"Rank {rank} reading bytes from {start} to {end - 1}")

with open(filename, 'r', encoding='utf-8') as f:
    # Move to start offset
    f.seek(start)

    # Skip partial line if not rank 0
    if start != 0:
        f.readline()

    current_position = f.tell()

    while current_position <= end:
        line = f.readline()
        num += 1
        if not line:
            break  # EOF
        # Process the line
        #print(f"Rank {rank} read: {line.strip()} END%^&**(*^%%$)")
        current_position = f.tell()

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

total_user_map = comm.gather(user_map, root = 0)
total_time_map = comm.gather(time_map, root = 0)

if rank == 0:
    top_5_users = heapq.nlargest(5, total_user_map[0].items(), key=lambda x: x[1])
    bottom_5_users = heapq.nsmallest(5, total_user_map[0].items(), key=lambda x: x[1])
    top_5_times = heapq.nlargest(5, total_time_map[0].items(), key=lambda x: x[1])
    bottom_5_time = heapq.nsmallest(5, total_time_map[0].items(), key=lambda x: x[1])
    print(top_5_users)
    print("******************")
    print(top_5_times)
    print("******************")
    print(bottom_5_users)
    print("******************")
    print(bottom_5_time)