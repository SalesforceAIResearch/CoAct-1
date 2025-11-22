from pathlib import Path
import sys

import docker.models
import docker.models.containers
sys.path.append(Path(__file__).parent.parent.parent.parent.as_posix())
import asyncio
if sys.platform == 'win32':
	asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import logging
import os
import platform
import time
import docker
import psutil
import requests
from filelock import FileLock
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List, Set
from desktop_env.providers.docker.manager import DockerVMManager
import traceback
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed, retry_if_exception_type
from tenacity.retry import retry_if_not_result
import aiohttp  # For async HTTP requests
from concurrent.futures import ThreadPoolExecutor
import functools
# import aiodns
import cachetools
from contextlib import asynccontextmanager
import random

# Setup logging
logger = logging.getLogger("desktopenv.providers.docker.DockerProviderService")
logger.setLevel(logging.DEBUG)

# Add a console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Constants
WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 10

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    'stop': stop_after_attempt(5),
    # 'wait': wait_exponential(multiplier=1, min=1, max=10),
    'wait': wait_fixed(5),
    'retry': retry_if_exception_type((docker.errors.APIError, requests.exceptions.RequestException, 
                                     ConnectionError, TimeoutError)),
    'reraise': True,
    'before_sleep': tenacity.before_sleep_log(logger, logging.WARNING)
}

class VMStartConfig(BaseModel):
    headless: bool = True
    os_type: str
    name: Optional[str] = None  # New: optional container name
    disk_size: Optional[str] = None
    ram_size: Optional[str] = None
    cpu_cores: Optional[str] = None

class VMConfig(VMStartConfig):
    # Server-side complete VM configuration
    path_to_vm: str
    disk_size: str = "10G"
    ram_size: str = "1G"
    cpu_cores: str = "2"

class SnapshotConfig(BaseModel):
    snapshot_name: str
    container_name: str  # Add container name to the config

class PortAllocationError(Exception):
    pass

class ContainerInfo:
    """Helper class to store container-specific information"""
    def __init__(self):
        self.container: docker.models.containers.Container = None
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.vm_config = None

class DockerProviderService:
    def __init__(self):
        self.manager = DockerVMManager()
        self.client = docker.from_env()
        
        # Store multiple container instances
        self.containers: Dict[str, ContainerInfo] = {}
        
        # Replace file lock with asyncio lock for port allocation
        self.port_allocation_lock = asyncio.Lock()
        
        # Track allocated ports
        self.allocated_ports: Set[int] = set()

        # Default VM configurations for different OS types
        self.vm_templates = {
            "ubuntu": {
                "path_to_vm": self.manager.get_vm_path("Ubuntu", region=None),
                # "disk_size": "32G",
                # "ram_size": "1G",
                # "cpu_cores": "2"
            }
        }

        # Specialized thread pools for different types of operations
        self.io_executor = ThreadPoolExecutor(max_workers=20)  # For I/O-bound operations
        self.cpu_executor = ThreadPoolExecutor(max_workers=max(4, os.cpu_count() or 4))  # For CPU-bound tasks
        
        # In-memory cache for port information
        self.local_port_cache = set()
        
        # Shared aiohttp session for HTTP requests
        self.aiohttp_session = None
        
        # Semaphore to limit concurrent container operations
        max_concurrent_operations = 8 if not sys.platform == 'win32' else 2
        self.container_semaphore = asyncio.Semaphore(max_concurrent_operations)
        logger.info(f"\nContainer semaphore initialized with {max_concurrent_operations} permits\n Using docker_remote_fc_mode")

    async def startup(self):
        """Initialize resources that should be created in an async context"""
        if self.aiohttp_session is None:
            # Configure with connection pooling and DNS caching
            # resolver = aiodns.DNSResolver(loop=asyncio.get_event_loop())
            tcp_connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool size
                ttl_dns_cache=300,  # DNS cache TTL in seconds
                use_dns_cache=True,
                # resolver=resolver
            )
            self.aiohttp_session = aiohttp.ClientSession(connector=tcp_connector)

    async def shutdown(self):
        """Clean up resources when service is shutting down"""
        # Close executors
        self.io_executor.shutdown(wait=False)
        self.cpu_executor.shutdown(wait=False)
        
        # Close aiohttp session
        if self.aiohttp_session:
            await self.aiohttp_session.close()
            self.aiohttp_session = None

    def _generate_container_name(self) -> str:
        """Generate a unique container name"""
        import uuid
        while True:
            name = f"desktop-env-{uuid.uuid4().hex[:8]}"
            if name not in self.containers:
                return name

    async def _get_used_ports(self):
        """Get all currently used ports (both system and Docker) asynchronously."""
        # Include our internally allocated ports to prevent reuse
        ports = await asyncio.get_event_loop().run_in_executor(
            self.io_executor, 
            self._get_used_ports_sync
        )
        return ports | self.local_port_cache | self.allocated_ports
    
    def _get_used_ports_sync(self):
        """Synchronous version of get_used_ports to run in thread pool."""
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        docker_ports = set()
        for container in self.client.containers.list():
            ports = container.attrs['NetworkSettings']['Ports']
            if ports:
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p['HostPort']) for p in port_mappings)
        return system_ports | docker_ports

    @retry(**{
        **DEFAULT_RETRY_CONFIG,
        'retry': retry_if_exception_type((docker.errors.APIError, requests.exceptions.ConnectionError, PortAllocationError))
    })
    async def _get_available_port(self, start_port: int, end_port: int) -> int:
        """Find next available port starting from start_port with retry."""
        try:
            used_ports = await self._get_used_ports()
            port = start_port
            while port < end_port:
                if port not in used_ports:
                    # Mark this port as allocated to prevent other concurrent allocations
                    self.allocated_ports.add(port)
                    self.local_port_cache.add(port)
                    return port
                await asyncio.sleep(0.01)  # Small sleep to avoid busy waiting
                port += 1
            raise PortAllocationError(f"No available ports found starting from {start_port}")
        except Exception as e:
            logger.error(f"Error finding available port: {str(e)}")
            raise

    def _validate_and_prepare_vm_config(self, start_config: VMStartConfig, container_name: str):
        """Validate and prepare VM configuration based on OS type."""
        if start_config.os_type.lower() not in self.vm_templates:
            raise ValueError(f"Unsupported OS type: {start_config.os_type}")
        
        template = self.vm_templates[start_config.os_type.lower()]
        
        # Create a dictionary with template values
        config_dict = dict(template)
        
        # Override with user-provided values if they exist
        for param in ['disk_size', 'ram_size', 'cpu_cores']:
            user_value = getattr(start_config, param)
            if user_value is not None:
                config_dict[param] = user_value
        
        # Add required parameters from start_config
        config_dict['headless'] = start_config.headless
        config_dict['os_type'] = start_config.os_type
        
        container_info = ContainerInfo()
        container_info.vm_config = VMConfig(**config_dict)
        
        self.containers[container_name] = container_info
        return container_info.vm_config

    @retry(**{
        **DEFAULT_RETRY_CONFIG,
        'retry': retry_if_exception_type((docker.errors.APIError, requests.exceptions.ConnectionError, PortAllocationError))
    })
    async def start_container(self, start_config: VMStartConfig):
        """
        Start a Docker container with retry logic.
        In the fc version, we separate `start_emulator` and `start_container`.
        """
        # Ensure aiohttp session is initialized
        await self.startup()
        
        container_name = start_config.name or self._generate_container_name()
        if (container_name in self.containers):
            raise ValueError(f"Container with name '{container_name}' already exists")

        logger.info(f"Starting container '{container_name}'...")
        
        # Prepare full VM configuration
        config = self._validate_and_prepare_vm_config(start_config, container_name)
        container_info = self.containers[container_name]
        
        # Use semaphore to limit concurrent container operations
        async with self.container_semaphore:
            try:
                allocated_ports = []
                
                # Use asyncio lock instead of file lock for port allocation
                async with self.port_allocation_lock:
                    logger.debug(f"Port allocation lock acquired for container {container_name}")
                    
                    # Allocate all ports in parallel
                    port_tasks = [
                        self._get_available_port(8006, 8193),  # VNC
                        self._get_available_port(5000, 5300),  # Server
                        self._get_available_port(9222, 9522),  # Chromium
                        self._get_available_port(8194, 8380),  # VLC
                    ]
                    vnc_port, server_port, chromium_port, vlc_port = await asyncio.gather(*port_tasks)
                    allocated_ports = [vnc_port, server_port, chromium_port, vlc_port]
                    
                    # Log allocated ports
                    logger.debug(f"Reserved ports: VNC:{vnc_port}, Server:{server_port}, Chrome:{chromium_port}, VLC:{vlc_port}")
                
                # Store allocated ports in container info - outside the lock
                container_info.vnc_port = vnc_port
                container_info.server_port = server_port
                container_info.chromium_port = chromium_port
                container_info.vlc_port = vlc_port

                # Prepare environment for Docker container
                environment = {
                    "DISK_SIZE": config.disk_size,
                    "RAM_SIZE": config.ram_size,
                    "CPU_CORES": config.cpu_cores
                }
                logger.info(f"Container '{container_name}' environment: {environment}")

                # Run Docker operations in IO thread pool
                # # Add a random delay before container creation to help with resource contention
                # delay_seconds = random.uniform(0.5, 10.0)
                # logger.info(f"Adding random delay of {delay_seconds:.2f}s before creating container '{container_name}'")
                # await asyncio.sleep(delay_seconds)
                
                def run_container():
                    """
                    equivalent to:
                    docker run -it --rm --entrypoint /bin/bash -e "DISK_SIZE=64G" -e "RAM_SIZE=8G" -e "CPU_CORES=8" --volume "/path/to/Ubuntu.qcow2:/System.qcow2:ro" --cap-add NET_ADMIN --device /dev/kvm -p 8007:8006 -p 5001:5000 happysixd/osworld-docker -c "sleep infinity"
                    """
                    return self.client.containers.run(
                        "happysixd/osworld-docker",
                        command=["-c", "sleep infinity"],
                        entrypoint="/bin/bash",
                        environment=environment,
                        name=container_name,
                        cap_add=["NET_ADMIN"],
                        devices=["/dev/kvm"],
                        volumes={
                            os.path.abspath(config.path_to_vm): {
                                "bind": "/System.qcow2",
                                "mode": "ro"
                            },
                            # os.path.abspath(Path(__file__).parent / "entry.sh"): {
                            #     "bind": "/run/entry.sh",
                            #     "mode": "ro"
                            # }
                        },
                        ports={
                            8006: container_info.vnc_port,
                            5000: container_info.server_port,
                            9222: container_info.chromium_port,
                            8080: container_info.vlc_port
                        },
                        detach=True
                    )
                
                # Add timeout and retry logic to the executor call
                @retry(
                    stop=stop_after_attempt(4),
                    retry=retry_if_exception_type((docker.errors.APIError, requests.exceptions.RequestException, 
                                               ConnectionError, TimeoutError, asyncio.TimeoutError)),
                    reraise=True,
                    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
                )
                async def run_with_timeout_and_retry():
                    try:
                        # Run with timeout
                        return await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(self.io_executor, run_container),
                            timeout=45  # 45s timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"Container start operation for '{container_name}' timed out after 120 seconds")
                        raise
                
                try:
                    container_info.container = await run_with_timeout_and_retry()
                except tenacity.RetryError as e:
                    logger.error(f"Failed to start container '{container_name}' after 4 attempts: {str(e.last_attempt.exception())}")
                    # Clean up allocated ports
                    if allocated_ports:
                        for port in allocated_ports:
                            if port in self.local_port_cache:
                                self.local_port_cache.remove(port)
                    raise

            
                logger.info(f"Started container '{container_name}' with ports - VNC: {container_info.vnc_port}, "
                            f"Server: {container_info.server_port}, Chrome: {container_info.chromium_port}, "
                            f"VLC: {container_info.vlc_port}")
                # do not check immediately, wait for a while; to avoid two much requests to the server
                # in fc, we do not need to wait for the VM to be ready, we start the emulator later
                # await asyncio.sleep(3)
                # await self._wait_for_vm_ready(container_name)
                return {"name": container_name, "connection_info": self.get_connection_info(container_name)}

            except Exception as e:
                # If container creation failed, clear the ports from the cache
                if self.local_port_cache and allocated_ports:
                    for port in allocated_ports:
                        if port in self.local_port_cache:
                            self.local_port_cache.remove(port)
                    logger.debug(f"Released allocated ports after failure: {allocated_ports}")
                
                logger.error(f"Error starting container '{container_name}': {str(e)}")
                if container_name in self.containers:
                    try:
                        await self.stop_container(container_name)
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup after failed start: {str(cleanup_error)}")
                raise

    @retry(**{
        'stop': stop_after_attempt(12),  # Reduce attempts to stop after ~60 seconds (12 attempts × 5 seconds)
        'wait': wait_fixed(5),  # Fixed wait time of 5 seconds
        'retry': retry_if_not_result(lambda result: result is True),
        'reraise': True,
        'before_sleep': tenacity.before_sleep_log(logger, logging.INFO)  # Log before retry attempts
    })
    async def _wait_for_vm_ready(self, container_name: str):
        """Wait for VM to be ready with exponential backoff retry logic."""
        logger.info(f"Checking if virtual machine '{container_name}' is ready...")
        return await self._is_vm_ready(container_name)

    @retry(**{
        **DEFAULT_RETRY_CONFIG,
        'retry': retry_if_exception_type(Exception)
    })
    async def start_emulator(self, container_name: str):
        """Start a Docker container with retry logic."""
        if container_name not in self.containers:
            raise ValueError(f"Container '{container_name}' not found")
            
        container_info = self.containers[container_name]
        
        logger.info(f"Starting emulator in container '{container_name}'...")

        async with self.container_semaphore:
            try:
                # Run Docker exec commands in IO thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self.io_executor,
                    lambda: (
                        container_info.container.exec_run(
                        # ["bash", "/run/entry.sh"],
                        ["bash", "-c", "/run/entry.sh > /tmp/entry_output.log 2>&1 &"],
                        detach=True
                    )
                ))

                # 等待一段时间，让命令产生一些输出
                await asyncio.sleep(4)
                
                # # 读取初始输出日志
                # log_result_prepare_env = await asyncio.get_event_loop().run_in_executor(
                #     self.io_executor,
                #     lambda: container_info.container.exec_run(
                #         ["cat", "/tmp/prepare_env.log"]
                #     )
                # )
                # log_result_qemu = await asyncio.get_event_loop().run_in_executor(
                #     self.io_executor,
                #     lambda: container_info.container.exec_run(
                #         ["cat", "/tmp/qemu.log"]
                #     )
                # )
                # output_prepare_env, output_qemu = log_result_prepare_env[1], log_result_qemu[1]
                # logger.info(f"Start emulator command for {container_name} prepare_env output: {output_prepare_env.decode('utf-8')}, qemu output: {output_qemu.decode('utf-8')}")

                # 读取初始输出日志
                log_result = await asyncio.get_event_loop().run_in_executor(
                    self.io_executor,
                    lambda: container_info.container.exec_run(
                        ["cat", "/tmp/entry_output.log"]
                    )
                )
                
                _, output = log_result
                logger.info(f"Start emulator command for {container_name} output: ")
                logger.info(output.decode('utf-8'))

                # Wait for VM to become ready
                try:
                    await self._wait_for_vm_ready(container_name)
                except tenacity.RetryError:
                    # If _wait_for_vm_ready exceeded its retry attempts (100 seconds)
                    logger.error(f"VM failed to become ready for container '{container_name}' after timeout, restarting the container...")
                    # directly restart the container
                    await asyncio.get_event_loop().run_in_executor(
                        self.io_executor,
                        container_info.container.restart
                    )
                    # Raise our custom error to trigger retry in start_emulator
                    raise TimeoutError(f"VM failed to become ready for container '{container_name}'")
                
                logger.info(f"Emulator started successfully in container '{container_name}'")
                return {"status": "success", "connection_info": self.get_connection_info(container_name)}
            
            except Exception as e:
                logger.error(f"Error starting emulator in container '{container_name}': {str(e)}")
                raise

    async def _is_vm_ready(self, container_name: str) -> bool:
        """Check if VM is ready by checking screenshot endpoint asynchronously."""
        container_info = self.containers[container_name]
        try:
            # Use the shared aiohttp session
            async with self.aiohttp_session.get(
                f"http://localhost:{container_info.server_port}/screenshot",
                timeout=10
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"VM not ready yet: {str(e)}")
            return False

    def get_connection_info(self, container_name: str):
        if container_name not in self.containers:
            raise ValueError(f"Container '{container_name}' not found")
            
        container_info = self.containers[container_name]
        if not all([container_info.server_port, container_info.chromium_port, 
                   container_info.vnc_port, container_info.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
            
        return {
            "server_port": container_info.server_port,
            "chromium_port": container_info.chromium_port,
            "vnc_port": container_info.vnc_port,
            "vlc_port": container_info.vlc_port
        }

    @retry(**{
        **DEFAULT_RETRY_CONFIG,
        'stop': stop_after_attempt(3),
        'retry': retry_if_exception_type((docker.errors.APIError, requests.exceptions.ConnectionError))
    })
    async def stop_container(self, container_name: str):
        """Stop a Docker container with retry logic."""
        if container_name not in self.containers:
            logger.info(f"Container '{container_name}' not found for stopping")
            return {"status": "success", "message": f"Container '{container_name}' not found"}

        # Use semaphore to limit concurrent container operations
        async with self.container_semaphore:
            container_info = self.containers[container_name]
            if container_info.container:
                logger.info(f"Stopping container '{container_name}'...")
                try:
                    # Run Docker operations in IO thread pool
                    await asyncio.get_event_loop().run_in_executor(
                        self.io_executor,
                        lambda: container_info.container.stop(timeout=30)
                    )
                    
                    await asyncio.get_event_loop().run_in_executor(
                        self.io_executor,
                        container_info.container.remove
                    )
                    
                    await asyncio.sleep(WAIT_TIME)  # Use asyncio.sleep instead of time.sleep
                    
                    # Release allocated ports
                    used_ports = [container_info.server_port, container_info.chromium_port, container_info.vnc_port, container_info.vlc_port]
                    
                    # Use lock to safely update shared port collections
                    async with self.port_allocation_lock:
                        for port in used_ports:
                            if port in self.allocated_ports:
                                self.allocated_ports.remove(port)
                            if port in self.local_port_cache:
                                self.local_port_cache.remove(port)
                    
                    del self.containers[container_name]
                    return {"status": "success", "message": f"Container '{container_name}' stopped successfully"}
                except Exception as e:
                    logger.error(f"Error stopping container '{container_name}': {str(e)}")
                    raise
            
            del self.containers[container_name]
            return {"status": "success", "message": f"Container '{container_name}' stopped successfully"}

    @retry(**{
        **DEFAULT_RETRY_CONFIG,
        'retry': retry_if_exception_type((docker.errors.APIError, requests.exceptions.ConnectionError, PortAllocationError))
    })
    async def stop_emulator(self, container_name: str):
        """Stop a Docker container with retry logic."""
        if container_name not in self.containers:
            raise ValueError(f"Container '{container_name}' not found")
            
        container_info = self.containers[container_name]
        
        logger.info(f"Stopping emulator in container '{container_name}'...")
        
        try:
            # directly restart the container
            await asyncio.get_event_loop().run_in_executor(
                self.io_executor,
                container_info.container.restart
            )
            await asyncio.sleep(3)  # Wait for a while to ensure the emulator is stopped
            logger.info(f"Emulator stopped successfully in container '{container_name}'")
            return {"status": "success", "message": f"Emulator stopped successfully in container '{container_name}'"}
            
        except Exception as e:
            logger.error(f"Error stopping emulator in container '{container_name}': {str(e)}")
            raise

    @retry(**{
        **DEFAULT_RETRY_CONFIG,
        'stop': stop_after_attempt(3),
        'retry': retry_if_exception_type((docker.errors.APIError, requests.exceptions.ConnectionError))
    })
    async def revert_to_snapshot(self, container_name: str, snapshot_name: str):
        """Revert to snapshot for a specific container with retry logic."""
        # if container_name not in self.containers:
        #     raise ValueError(f"Container '{container_name}' not found")
        logger.info(f"Reverting container '{container_name}' to snapshot '{snapshot_name}'")
        return await self.stop_emulator(container_name)

# FastAPI application
docker_service = DockerProviderService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    await docker_service.startup()
    yield
    # Shutdown: clean up resources
    await docker_service.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/start_container")
async def start_container(config: VMStartConfig):
    try:
        result = await docker_service.start_container(config)
        return {"status": "success", **result}
    except tenacity.RetryError as e:
        logger.error(f"Failed to start container after multiple retries: {str(e.last_attempt.exception())}")
        raise HTTPException(status_code=500, 
                           detail=f"Failed to start container after multiple retries: {str(e.last_attempt.exception())}")
    except Exception as e:
        logger.error(f"Error starting container: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_container/{container_name}")
async def stop_container(container_name: str):
    try:
        result = await docker_service.stop_container(container_name)
        return {"status": "success", **result}
    except tenacity.RetryError as e:
        logger.error(f"Failed to stop container after multiple retries: {str(e.last_attempt.exception())}")
        raise HTTPException(status_code=500, 
                           detail=f"Failed to stop container after multiple retries: {str(e.last_attempt.exception())}")
    except Exception as e:
        logger.error(f"Error stopping container: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start/{container_name}")
async def start_vm(container_name: str):
    try:
        return await docker_service.start_emulator(container_name)
    except tenacity.RetryError as e:
        logger.error(f"Failed to start VM after multiple retries: {str(e.last_attempt.exception())}")
        raise HTTPException(status_code=500, 
                           detail=f"Failed to start VM after multiple retries: {str(e.last_attempt.exception())}")
    except Exception as e:
        logger.error(f"Error starting VM: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop/{container_name}")
async def stop_vm(container_name: str):
    try:
        return await docker_service.stop_emulator(container_name)
    except tenacity.RetryError as e:
        logger.error(f"Failed to stop VM after multiple retries: {str(e.last_attempt.exception())}")
        raise HTTPException(status_code=500, 
                           detail=f"Failed to stop VM after multiple retries: {str(e.last_attempt.exception())}")
    except Exception as e:
        logger.error(f"Error stopping VM: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/snapshot/revert")
async def revert_vm_snapshot(config: SnapshotConfig):
    try:
        return await docker_service.revert_to_snapshot(config.container_name, config.snapshot_name)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{container_name}")
async def get_status(container_name: str):
    try:
        if container_name in docker_service.containers:
            return {
                "status": "running",
                "connection_info": docker_service.get_connection_info(container_name)
            }
        return {"status": "stopped"}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list")
async def list_containers():
    """List all running containers"""
    return {
        "containers": [
            {
                "name": name,
                "connection_info": docker_service.get_connection_info(name)
            }
            for name in docker_service.containers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7766)
