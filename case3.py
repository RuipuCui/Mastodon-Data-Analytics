import json
import heapq
from datetime import datetime
from mpi4py import MPI
import os
import time as t

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

filename = "../mastodon-16m.ndjson"

# Start the timer
timer = t.time()

# Calculate chunk offsets for each process
if rank == 0:
    file_size = os.path.getsize(filename)
    chunk_size = file_size // size
    start = rank * chunk_size
    end = file_size if rank == size - 1 else (rank + 1) * chunk_size
else:
    start = None
    end = None

# Broadcast start and end positions to all processes
start = comm.bcast(start, root=0)
end = comm.bcast(end, root=0)

# Dictionaries to store local sentiment data for each process
user_map = {}  # user_id$username -> total sentiment
time_map = {}  # hour timestamp -> total sentiment

# Each process reads and processes its assigned portion of the file
with open(filename, 'r', encoding='utf-8') as f:
    f.seek(start)

    # If not the first process, skip the partial line to ensure full line reads
    if start != 0:
        f.readline()

    current_position = f.tell()

    # Read and process lines within this process's assigned byte range
    while current_position < end:
        line = f.readline()
        if not line:
            break  # Reached EOF
        current_position = f.tell()

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # Skip malformed JSON lines

        # Extract necessary fields
        username = item.get("doc", {}).get("account", {}).get("username")
        user_id = item.get("doc", {}).get("account", {}).get("id")
        sentiment = item.get("doc", {}).get("sentiment")
        createdAt = item.get("doc", {}).get("createdAt")

        # Skip entries with missing information
        if user_id is None or username is None or sentiment is None or createdAt is None:
            continue

        try:
            dt = datetime.fromisoformat(createdAt.replace("Z", "+00:00"))
        except ValueError:
            continue  # Skip if datetime parsing fails

        # Aggregate by hour
        time = f"{dt.date().isoformat()} {dt.hour:02d}"

        # Combine user ID and username to ensure uniqueness
        user_info = str(user_id) + '$' + username

        # Aggregate sentiment per user
        if user_info in user_map:
            user_map[user_info] += sentiment
        else:
            user_map[user_info] = sentiment

        # Aggregate sentiment per time period
        if time in time_map:
            time_map[time] += sentiment
        else:
            time_map[time] = sentiment

# Gather all user and time maps to rank 0 process
all_user_map = comm.gather(user_map, root=0)
all_time_map = comm.gather(time_map, root=0)

# Only the root process (rank 0) processes and prints the final result
if rank == 0:
    total_user_map = {}
    total_time_map = {}

    # Merge user maps from all processes
    for process_user_map in all_user_map:
        for user, sentiment in process_user_map.items():
            total_user_map[user] = total_user_map.get(user, 0) + sentiment

    # Merge time maps from all processes
    for process_time_map in all_time_map:
        for time_key, sentiment in process_time_map.items():
            total_time_map[time_key] = total_time_map.get(time_key, 0) + sentiment

    # Get top/bottom 5 users and time periods by sentiment
    top_5_users = heapq.nlargest(5, total_user_map.items(), key=lambda x: x[1])
    bottom_5_users = heapq.nsmallest(5, total_user_map.items(), key=lambda x: x[1])
    top_5_times = heapq.nlargest(5, total_time_map.items(), key=lambda x: x[1])
    bottom_5_times = heapq.nsmallest(5, total_time_map.items(), key=lambda x: x[1])

    # End timing
    run_time = t.time() - timer

    # Output results
    print("case 3: 2 nodes, 4 cores each")
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
