#!/usr/bin/env python3
"""
gRPC-based inference server main entry point
"""

import sys
import os
from loguru import logger
from grpc_server import serve

def main():
    """Main entry point for gRPC inference server"""
    
    # Configure logging (same as original main.py)
    logger.remove()  # Remove default handler
    logger.add(
        "logs/runtime_{time}.log",
        rotation="2 GB",
        retention="7 days", 
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
        level="DEBUG",
        enqueue=True
    )
    
    logger.info("starting grpc inference server...")
    
    try:
        serve()
    except Exception as e:
        logger.error(f"failed to start grpc server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
