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

print(f"Rank {rank} reading bytes from {start} to {end - 1}")

with open(filename, 'r', encoding='utf-8') as f:
    # Move to start offset
    f.seek(start)

    # Skip partial line if not rank 0
    if start != 0:
        f.readline()

    current_position = f.tell()

    while current_position < end:
        line = f.readline()
        num += 1
        if not line:
            break  # EOF
        # Process the line
        #print(f"Rank {rank} read: {line.strip()} END%^&**(*^%%$)")
        current_position = f.tell()

print(num)