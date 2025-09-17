#!/usr/bin/env python3
"""
Concurrent stress test for cam-stream server
Tests for race conditions, nil pointer exceptions, and concurrent access issues
"""
import json
import random
import threading
import time
import requests
import uuid
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import queue
import signal

# Add parent directories to path for logging
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from loguru import logger

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}:{line}</cyan> | {message}",
    level="INFO"
)

# Test configuration
SERVER_BASE_URL = "http://localhost:8080"
CONFIG_FILE_PATH = "../cam-stream/configs/config.json"
MAX_CONCURRENT_THREADS = 20
TEST_DURATION_SECONDS = 60
REQUEST_TIMEOUT = 5

# Statistics tracking
class TestStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.error_count = 0
        self.nil_pointer_errors = 0
        self.race_condition_errors = 0
        self.timeout_errors = 0
        self.errors = []
        
    def record_success(self):
        with self.lock:
            self.total_requests += 1
            self.successful_requests += 1
    
    def record_failure(self, error_type: str, error_msg: str):
        with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.error_count += 1
            self.errors.append((error_type, error_msg, time.time()))
            
            # Categorize errors
            if "nil pointer" in error_msg.lower() or "null pointer" in error_msg.lower():
                self.nil_pointer_errors += 1
            elif "race" in error_msg.lower() or "concurrent" in error_msg.lower():
                self.race_condition_errors += 1
            elif "timeout" in error_msg.lower():
                self.timeout_errors += 1
    
    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "error_count": self.error_count,
                "nil_pointer_errors": self.nil_pointer_errors,
                "race_condition_errors": self.race_condition_errors,
                "timeout_errors": self.timeout_errors,
                "success_rate": (self.successful_requests / max(1, self.total_requests)) * 100
            }


class CamStreamTester:
    def __init__(self, base_url: str, stats: TestStats):
        self.base_url = base_url
        self.stats = stats
        self.session = requests.Session()
        self.session.timeout = REQUEST_TIMEOUT
        self.camera_ids = set()
        self.inference_server_ids = set()
        self.camera_lock = threading.Lock()
        self.server_lock = threading.Lock()
        
    def generate_random_camera_config(self) -> Dict:
        """generate random camera configuration"""
        camera_id = f"test_cam_{uuid.uuid4().hex[:8]}"
        rtsp_urls = [
            "rtsp://admin:password@192.168.1.100:554/stream1",
            "rtsp://user:pass@10.0.0.50:554/live",
            "rtsp://cam:123456@172.16.1.200:554/main",
            "rtsp://test:test@127.0.0.1:8554/test"
        ]
        
        return {
            "id": camera_id,
            "name": f"Test Camera {random.randint(1, 1000)}",
            "rtsp_url": random.choice(rtsp_urls),
            "enabled": random.choice([True, False]),
            "running": random.choice([True, False]),
            "inference_server_bindings": self.generate_random_bindings()
        }
    
    def generate_random_inference_server_config(self) -> Dict:
        """generate random inference server configuration"""
        server_id = f"test_inf_{uuid.uuid4().hex[:8]}"
        model_types = ["yolo", "detectron2", "custom", "fall", "gesture", "tshirt"]
        
        return {
            "id": server_id,
            "name": f"Test Inference Server {random.randint(1, 1000)}",
            "url": f"http://localhost:{random.randint(8900, 8999)}",
            "model_type": random.choice(model_types),
            "description": f"Test server for {random.choice(model_types)} detection",
            "enabled": random.choice([True, False])
        }
    
    def generate_random_bindings(self) -> List[Dict]:
        """generate random server bindings for cameras"""
        num_bindings = random.randint(0, 3)
        bindings = []
        
        for _ in range(num_bindings):
            bindings.append({
                "server_id": f"server_{uuid.uuid4().hex[:8]}",
                "threshold": random.uniform(0.1, 0.9),
                "max_threshold": random.uniform(0.5, 1.0)
            })
        
        return bindings
    
    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[requests.Response]:
        """make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                response = self.session.post(url, json=data)
            elif method == "PUT":
                response = self.session.put(url, json=data)
            elif method == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"unsupported HTTP method: {method}")
            
            return response
            
        except requests.exceptions.Timeout:
            self.stats.record_failure("timeout", f"timeout on {method} {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            self.stats.record_failure("connection", f"connection error on {method} {endpoint}")
            return None
        except Exception as e:
            self.stats.record_failure("request", f"request error on {method} {endpoint}: {str(e)}")
            return None
    
    def create_camera(self):
        """create a new camera"""
        camera_config = self.generate_random_camera_config()
        response = self.make_request("POST", "/api/cameras", camera_config)
        
        if response and response.status_code in [200, 201]:
            try:
                result = response.json()
                if result.get("success"):
                    with self.camera_lock:
                        self.camera_ids.add(camera_config["id"])
                    self.stats.record_success()
                    logger.debug(f"created camera: {camera_config['id']}")
                else:
                    self.stats.record_failure("api_error", f"camera creation failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for camera creation")
        elif response:
            self.stats.record_failure("http_error", f"camera creation HTTP {response.status_code}: {response.text}")
    
    def update_camera(self):
        """update an existing camera"""
        with self.camera_lock:
            if not self.camera_ids:
                return
            camera_id = random.choice(list(self.camera_ids))
        
        updated_config = self.generate_random_camera_config()
        response = self.make_request("PUT", f"/api/cameras/{camera_id}", updated_config)
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    self.stats.record_success()
                    logger.debug(f"updated camera: {camera_id}")
                else:
                    self.stats.record_failure("api_error", f"camera update failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for camera update")
        elif response:
            self.stats.record_failure("http_error", f"camera update HTTP {response.status_code}: {response.text}")
    
    def delete_camera(self):
        """delete an existing camera"""
        with self.camera_lock:
            if not self.camera_ids:
                return
            camera_id = random.choice(list(self.camera_ids))
            self.camera_ids.discard(camera_id)
        
        response = self.make_request("DELETE", f"/api/cameras/{camera_id}")
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    self.stats.record_success()
                    logger.debug(f"deleted camera: {camera_id}")
                else:
                    self.stats.record_failure("api_error", f"camera deletion failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for camera deletion")
        elif response:
            self.stats.record_failure("http_error", f"camera deletion HTTP {response.status_code}: {response.text}")
    
    def get_cameras(self):
        """get all cameras"""
        response = self.make_request("GET", "/api/cameras")
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    self.stats.record_success()
                    logger.debug(f"retrieved {len(result.get('data', []))} cameras")
                else:
                    self.stats.record_failure("api_error", f"camera retrieval failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for camera retrieval")
        elif response:
            self.stats.record_failure("http_error", f"camera retrieval HTTP {response.status_code}: {response.text}")
    
    def create_inference_server(self):
        """create a new inference server"""
        server_config = self.generate_random_inference_server_config()
        response = self.make_request("POST", "/api/inference-servers", server_config)
        
        if response and response.status_code in [200, 201]:
            try:
                result = response.json()
                if result.get("success"):
                    with self.server_lock:
                        self.inference_server_ids.add(server_config["id"])
                    self.stats.record_success()
                    logger.debug(f"created inference server: {server_config['id']}")
                else:
                    self.stats.record_failure("api_error", f"inference server creation failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for inference server creation")
        elif response:
            self.stats.record_failure("http_error", f"inference server creation HTTP {response.status_code}: {response.text}")
    
    def update_inference_server(self):
        """update an existing inference server"""
        with self.server_lock:
            if not self.inference_server_ids:
                return
            server_id = random.choice(list(self.inference_server_ids))
        
        updated_config = self.generate_random_inference_server_config()
        response = self.make_request("PUT", f"/api/inference-servers/{server_id}", updated_config)
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    self.stats.record_success()
                    logger.debug(f"updated inference server: {server_id}")
                else:
                    self.stats.record_failure("api_error", f"inference server update failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for inference server update")
        elif response:
            self.stats.record_failure("http_error", f"inference server update HTTP {response.status_code}: {response.text}")
    
    def delete_inference_server(self):
        """delete an existing inference server"""
        with self.server_lock:
            if not self.inference_server_ids:
                return
            server_id = random.choice(list(self.inference_server_ids))
            self.inference_server_ids.discard(server_id)
        
        response = self.make_request("DELETE", f"/api/inference-servers/{server_id}")
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    self.stats.record_success()
                    logger.debug(f"deleted inference server: {server_id}")
                else:
                    self.stats.record_failure("api_error", f"inference server deletion failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for inference server deletion")
        elif response:
            self.stats.record_failure("http_error", f"inference server deletion HTTP {response.status_code}: {response.text}")
    
    def get_inference_servers(self):
        """get all inference servers"""
        response = self.make_request("GET", "/api/inference-servers")
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    self.stats.record_success()
                    logger.debug(f"retrieved {len(result.get('data', []))} inference servers")
                else:
                    self.stats.record_failure("api_error", f"inference server retrieval failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for inference server retrieval")
        elif response:
            self.stats.record_failure("http_error", f"inference server retrieval HTTP {response.status_code}: {response.text}")
    
    def test_config_operations(self):
        """test config file operations using separate test file"""
        try:
            # Create test config file
            test_config = {
                "test_mode": True,
                "test_frame_rate": random.randint(10, 120),
                "test_width": random.choice([1280, 1920, 3840]),
                "test_height": random.choice([720, 1080, 2160])
            }
            
            # Write to test config file in tools/test directory
            test_config_path = os.path.join(os.path.dirname(__file__), "test_config.json")
            with open(test_config_path, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            # Read it back to simulate config loading
            with open(test_config_path, 'r') as f:
                loaded_config = json.load(f)
            
            self.stats.record_success()
            logger.debug(f"test config operation completed: {test_config}")
            
        except Exception as e:
            self.stats.record_failure("config_test_error", f"test config operation failed: {str(e)}")
    
    def get_status(self):
        """get server status"""
        response = self.make_request("GET", "/api/status")
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    self.stats.record_success()
                    logger.debug("retrieved server status")
                else:
                    self.stats.record_failure("api_error", f"status retrieval failed: {result.get('error', 'unknown')}")
            except json.JSONDecodeError:
                self.stats.record_failure("json_error", "invalid JSON response for status retrieval")
        elif response:
            self.stats.record_failure("http_error", f"status retrieval HTTP {response.status_code}: {response.text}")


def worker_thread(tester: CamStreamTester, stop_event: threading.Event, thread_id: int):
    """worker thread that performs random operations"""
    operations = [
        tester.create_camera,
        tester.update_camera,
        tester.delete_camera,
        tester.get_cameras,
        tester.create_inference_server,
        tester.update_inference_server,
        tester.delete_inference_server,
        tester.get_inference_servers,
        tester.test_config_operations,
        tester.get_status
    ]
    
    logger.info(f"worker thread {thread_id} started")
    
    while not stop_event.is_set():
        # Choose random operation with weighted probabilities
        weights = [15, 10, 5, 20, 15, 10, 5, 20, 5, 15]  # Higher weight = more frequent
        operation = random.choices(operations, weights=weights)[0]
        
        try:
            operation()
            # Random sleep to vary timing
            time.sleep(random.uniform(0.1, 1.0))
        except Exception as e:
            tester.stats.record_failure("worker_error", f"worker {thread_id} error: {str(e)}")
    
    logger.info(f"worker thread {thread_id} stopped")


def print_stats(stats: TestStats):
    """print current statistics"""
    current_stats = stats.get_stats()
    
    print(f"\n{'='*60}")
    print(f"CONCURRENCY TEST STATISTICS")
    print(f"{'='*60}")
    print(f"Total Requests:        {current_stats['total_requests']}")
    print(f"Successful Requests:   {current_stats['successful_requests']}")
    print(f"Failed Requests:       {current_stats['failed_requests']}")
    print(f"Success Rate:          {current_stats['success_rate']:.2f}%")
    print(f"")
    print(f"ERROR ANALYSIS:")
    print(f"Total Errors:          {current_stats['error_count']}")
    print(f"Nil Pointer Errors:    {current_stats['nil_pointer_errors']} 🚨")
    print(f"Race Condition Errors: {current_stats['race_condition_errors']} ⚠️")
    print(f"Timeout Errors:        {current_stats['timeout_errors']}")
    print(f"")
    
    # Show recent errors
    if stats.errors:
        print(f"RECENT ERRORS (last 5):")
        recent_errors = stats.errors[-5:]
        for error_type, error_msg, timestamp in recent_errors:
            print(f"  [{time.strftime('%H:%M:%S', time.localtime(timestamp))}] {error_type}: {error_msg}")


def signal_handler(signum, frame):
    """handle Ctrl+C gracefully"""
    logger.warning("received interrupt signal, stopping test...")
    global should_stop
    should_stop = True


def main():
    global should_stop
    should_stop = False
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 starting cam-stream concurrency stress test")
    logger.info(f"server: {SERVER_BASE_URL}")
    logger.info(f"threads: {MAX_CONCURRENT_THREADS}")
    logger.info(f"duration: {TEST_DURATION_SECONDS}s")
    
    # Initialize components
    stats = TestStats()
    tester = CamStreamTester(SERVER_BASE_URL, stats)
    stop_event = threading.Event()
    
    # Test server connectivity
    try:
        response = requests.get(f"{SERVER_BASE_URL}/api/ping", timeout=5)
        if response.status_code != 200:
            logger.error(f"server not responding correctly: HTTP {response.status_code}")
            return
        logger.success("server connectivity confirmed")
    except Exception as e:
        logger.error(f"cannot connect to server: {e}")
        logger.info("please ensure cam-stream server is running on http://localhost:8080")
        return
    
    # Start worker threads
    threads = []
    for i in range(MAX_CONCURRENT_THREADS):
        thread = threading.Thread(target=worker_thread, args=(tester, stop_event, i+1))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    logger.success(f"started {MAX_CONCURRENT_THREADS} worker threads")
    
    # Monitor test progress
    start_time = time.time()
    try:
        while not should_stop and (time.time() - start_time) < TEST_DURATION_SECONDS:
            time.sleep(5)  # Print stats every 5 seconds
            print_stats(stats)
            
    except KeyboardInterrupt:
        logger.warning("test interrupted by user")
    
    # Stop all threads
    logger.info("stopping all worker threads...")
    stop_event.set()
    
    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=2)
    
    # Final statistics
    print_stats(stats)
    
    final_stats = stats.get_stats()
    if final_stats['nil_pointer_errors'] > 0:
        logger.error(f"🚨 FOUND {final_stats['nil_pointer_errors']} NIL POINTER ERRORS!")
    
    if final_stats['race_condition_errors'] > 0:
        logger.warning(f"⚠️ FOUND {final_stats['race_condition_errors']} RACE CONDITION ERRORS!")
    
    if final_stats['error_count'] == 0:
        logger.success("✅ no concurrency issues detected!")
    elif final_stats['success_rate'] > 90:
        logger.info(f"✓ test completed with {final_stats['success_rate']:.1f}% success rate")
    else:
        logger.warning(f"⚠️ low success rate: {final_stats['success_rate']:.1f}%")
    
    logger.info("concurrency test completed")


if __name__ == "__main__":
    main()
