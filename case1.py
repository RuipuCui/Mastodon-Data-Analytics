import json
import heapq
from datetime import datetime
from mpi4py import MPI  # (Unused here, may be relevant if parallelizing later)
import os
import time as t

# Start a timer to measure runtime
timer = t.time()

# Dictionary to store total sentiment score per user
user_map = {}

# Dictionary to store total sentiment score per time period (hourly)
time_map = {}

# Open the Mastodon dataset file (assumes newline-delimited JSON format)
with open("../mastodon-144g.ndjson", "r") as f:
    for line in f:
        item = json.loads(line)

        # Extract relevant fields
        username = item.get("doc", {}).get("account", {}).get("username")
        user_id = item.get("doc", {}).get("account", {}).get("id")
        sentiment = item.get("doc", {}).get("sentiment")
        createdAt = item.get("doc", {}).get("createdAt")

        # Skip incomplete records
        if username is None or sentiment is None or createdAt is None:
            continue

        # Convert ISO date string to datetime object
        try:
            dt = datetime.fromisoformat(createdAt.replace("Z", "+00:00"))
        except ValueError:
            continue

        # Format time as "YYYY-MM-DD HH" for hourly aggregation
        time = f"{dt.date().isoformat()} {dt.hour:02d}"

        # Combine user_id and username to uniquely identify a user
        user_info = str(user_id) + '$' + username

        # Aggregate sentiment for the user
        if user_info in user_map:
            user_map[user_info] += sentiment
        else:
            user_map[user_info] = sentiment

        # Aggregate sentiment for the hourly time slot
        if time in time_map:
            time_map[time] += sentiment
        else:
            time_map[time] = sentiment

# Identify top and bottom 5 users based on sentiment score
top_5_users = heapq.nlargest(5, user_map.items(), key=lambda x: x[1])
bottom_5_users = heapq.nsmallest(5, user_map.items(), key=lambda x: x[1])

# Identify top and bottom 5 time periods based on sentiment score
top_5_times = heapq.nlargest(5, time_map.items(), key=lambda x: x[1])
bottom_5_times = heapq.nsmallest(5, time_map.items(), key=lambda x: x[1])

# Measure and display total runtime
run_time = t.time() - timer

print("case 1: 1 nodes, 1 cores")
print(f"\nProgram ran for: {run_time:.2f} seconds\n")

# Print top 5 users
print("***** TOP 5 USERS *****")
for user_info, sentiment in top_5_users:
    uid, uname = user_info.split('$', 1)
    print(f"User ID: {uid:<22} Username: {uname:<20} Total Sentiment: {sentiment:.4f}")

# Print top 5 time periods
print("\n***** TOP 5 TIMES *****")
for time_key, sentiment in top_5_times:
    print(f"Time: {time_key:<16} Total Sentiment: {sentiment:.4f}")

# Print bottom 5 users
print("\n***** BOTTOM 5 USERS *****")
for user_info, sentiment in bottom_5_users:
    uid, uname = user_info.split('$', 1)
    print(f"User ID: {uid:<22} Username: {uname:<20} Total Sentiment: {sentiment:.4f}")

# Print bottom 5 time periods
print("\n***** BOTTOM 5 TIMES *****")
for time_key, sentiment in bottom_5_times:
    print(f"Time: {time_key:<16} Total Sentiment: {sentiment:.4f}")






