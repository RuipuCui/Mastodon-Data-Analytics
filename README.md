#run
sbatch helloworld.slurm

#access data files 
#run those commands in your own home directory
ln -s /data/gpfs/projects/COMP90024/mastodon-106k.ndjson
ln -s /data/gpfs/projects/COMP90024/mastodon-16m.ndjson
ln -s /data/gpfs/projects/COMP90024/mastodon-144Gb.ndjson

cat ../mastodon-106k.ndjson