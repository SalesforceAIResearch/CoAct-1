This is the implementatin of docker_remote_fixed_containers(fc).

We start a fixed number of containers, and never destroy them. To reset the envs, we use `docker exec -it <container_name> xxx` to kill and restart the QEMU process.