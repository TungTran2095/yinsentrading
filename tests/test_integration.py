import unittest
import os
import sys
import requests
from unittest.mock import patch

class TestIntegration(unittest.TestCase):
    """Integration tests for Trading System"""
    
    def test_docker_compose_file(self):
        """Test that docker-compose.yml exists and is valid"""
        docker_compose_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker-compose.yml")
        self.assertTrue(os.path.exists(docker_compose_path), "docker-compose.yml should exist")
        
        # Check file size to ensure it's not empty
        self.assertGreater(os.path.getsize(docker_compose_path), 100, "docker-compose.yml should not be empty")
    
    def test_dockerfiles(self):
        """Test that Dockerfiles exist for all services"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        services = ["data_service", "model_service", "rl_service", "trading_service", "chat_service", "frontend"]
        
        for service in services:
            dockerfile_path = os.path.join(base_dir, service, "Dockerfile")
            self.assertTrue(os.path.exists(dockerfile_path), f"Dockerfile should exist for {service}")
            
            # Check file size to ensure it's not empty
            self.assertGreater(os.path.getsize(dockerfile_path), 50, f"Dockerfile for {service} should not be empty")
    
    def test_env_file(self):
        """Test that .env file exists and is valid"""
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        self.assertTrue(os.path.exists(env_path), ".env file should exist")
        
        # Check file size to ensure it's not empty
        self.assertGreater(os.path.getsize(env_path), 100, ".env file should not be empty")
    
    def test_nginx_config(self):
        """Test that nginx.conf exists and is valid"""
        nginx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "nginx.conf")
        self.assertTrue(os.path.exists(nginx_path), "nginx.conf should exist")
        
        # Check file size to ensure it's not empty
        self.assertGreater(os.path.getsize(nginx_path), 100, "nginx.conf should not be empty")
        
        # Check that it contains proxy configurations for all services
        with open(nginx_path, 'r') as f:
            content = f.read()
            self.assertIn("proxy_pass", content, "nginx.conf should contain proxy_pass directives")
            self.assertIn("data_service", content, "nginx.conf should proxy to data_service")
            self.assertIn("model_service", content, "nginx.conf should proxy to model_service")
            self.assertIn("trading_service", content, "nginx.conf should proxy to trading_service")
            self.assertIn("chat_service", content, "nginx.conf should proxy to chat_service")

if __name__ == '__main__':
    unittest.main()
