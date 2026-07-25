#!/usr/bin/env python3
"""
VLM Reward Client - Fetches VLM rewards via HTTP, designed for use inside Docker containers.

Usage:
    from vlm_reward_client import VLMRewardClient

    client = VLMRewardClient(server_url="http://host.docker.internal:5001")
    reward = client.compute_reward(image, task_description)
"""

import base64
from typing import List, Optional, Union

import numpy as np
import requests


class VLMRewardClient:
    """VLM Reward Client - communicates with the VLM Reward Server over HTTP."""
    
    def __init__(
        self,
        server_url: str = "http://host.docker.internal:5001",
        timeout: float = 30.0,
    ):
        """
        Initialize the VLM Reward Client.

        Args:
            server_url: Address of the VLM Reward Server.
                - From inside Docker to host: http://host.docker.internal:5001
                - On the same machine: http://localhost:5001
            timeout: Request timeout in seconds.
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
    def health_check(self) -> bool:
        """Check whether the server is available and the model is loaded."""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200 and response.json().get("model_loaded", False)
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def _encode_image(self, image: np.ndarray) -> dict:
        """Encode an image as a base64 dict with shape and dtype metadata."""
        image_bytes = image.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        return {
            "data": image_b64,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        }
    
    def compute_completion(
        self,
        image: np.ndarray,
        task_description: str,
    ) -> dict:
        """
        Determine whether the task is completed (yes/no).

        Returns:
            dict: {"score": 1.0 or 0.0, "completed": bool, "response": str, "success": bool}
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        image_bytes = image.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        request_data = {
            "image": image_b64,
            "image_shape": list(image.shape),
            "image_dtype": str(image.dtype),
            "task_description": task_description,
        }

        try:
            response = requests.post(
                f"{self.server_url}/compute_completion",
                json=request_data,
                timeout=self.timeout
            )
            result = response.json()
            if not result.get("success", False):
                return {"score": 0.0, "completed": False, "response": "", "success": False}
            return {
                "score": result["score"],
                "completed": result.get("completed", False),
                "response": result.get("response", ""),
                "success": True,
            }
        except Exception as e:
            print(f"VLM completion request failed: {e}")
            return {"score": 0.0, "completed": False, "response": "", "success": False}

    def compute_comparison(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        task_description: str,
    ) -> dict:
        """
        Compare two images to determine which is closer to task completion.

        Args:
            image_a: First observation image (earlier frame), shape (H, W, 3), dtype uint8
            image_b: Second observation image (later frame), shape (H, W, 3), dtype uint8
            task_description: Task description

        Returns:
            dict: {"result": "ImageA"/"ImageB"/None, "score": float, "response": str, "success": bool}
                  score: +1.0 = ImageB closer, -1.0 = ImageA closer, 0.0 = uncertain
        """
        if image_a.dtype != np.uint8:
            image_a = (image_a * 255).astype(np.uint8)
        if image_b.dtype != np.uint8:
            image_b = (image_b * 255).astype(np.uint8)

        request_data = {
            "image_a": base64.b64encode(image_a.tobytes()).decode('utf-8'),
            "image_a_shape": list(image_a.shape),
            "image_a_dtype": str(image_a.dtype),
            "image_b": base64.b64encode(image_b.tobytes()).decode('utf-8'),
            "image_b_shape": list(image_b.shape),
            "image_b_dtype": str(image_b.dtype),
            "task_description": task_description,
        }

        try:
            response = requests.post(
                f"{self.server_url}/compute_comparison",
                json=request_data,
                timeout=self.timeout
            )

            result = response.json()
            if not result.get("success", False):
                print(f"VLM comparison failed: {result.get('error', 'Unknown error')}")
                return {"result": None, "score": 0.0, "response": "", "success": False}

            return {
                "result": result["result"],
                "score": result["score"],
                "response": result.get("response", ""),
                "success": True,
            }
        except requests.exceptions.Timeout:
            print("VLM comparison request timed out")
            return {"result": None, "score": 0.0, "response": "", "success": False}
        except Exception as e:
            print(f"VLM comparison request failed: {e}")
            return {"result": None, "score": 0.0, "response": "", "success": False}

    def compute_roboreward(
        self,
        frames: list[np.ndarray],
        task_description: str,
        fps: float = 1.0,
    ) -> dict:
        """
        Score a trajectory via RoboReward (1-5 discrete, normalized to 0-1).

        Args:
            frames: RGB frame list, each shape=(H,W,3), dtype=uint8
            task_description: Task description
            fps: wall-clock frame rate of `frames` (control_freq / vlm_call_interval
                for the online training path).

        Returns:
            dict: {"score": float (0-1), "raw_score": int (1-5), "response": str, "success": bool}
        """
        encoded_frames = []
        for frame in frames:
            if frame is None:
                continue
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            encoded_frames.append(self._encode_image(frame))

        if not encoded_frames:
            return {"score": 0.0, "raw_score": None, "response": "", "success": False}

        request_data = {
            "frames": encoded_frames,
            "task_description": task_description,
            "fps": fps,
        }

        try:
            response = requests.post(
                f"{self.server_url}/compute_roboreward",
                json=request_data,
                timeout=self.timeout,
            )
            result = response.json()
            if not result.get("success", False):
                print(f"VLM roboreward failed: {result.get('error', 'Unknown error')}")
                return {"score": 0.0, "raw_score": None, "response": "", "success": False}
            return {
                "score": result["score"],
                "raw_score": result.get("raw_score"),
                "response": result.get("response", ""),
                "success": True,
            }
        except requests.exceptions.Timeout:
            print("VLM roboreward request timed out")
            return {"score": 0.0, "raw_score": None, "response": "", "success": False}
        except Exception as e:
            print(f"VLM roboreward request failed: {e}")
            return {"score": 0.0, "raw_score": None, "response": "", "success": False}

    def compute_topreward(
        self,
        frames: list[np.ndarray],
        task_description: str,
        fps: float = 2.0,
    ) -> dict:
        """
        Score a trajectory via TOPReward: log P("True" | video, instruction).

        Args:
            frames: RGB frame list, each shape=(H,W,3), dtype=uint8
            task_description: Task description
            fps: wall-clock frame rate of `frames` (control_freq / vlm_call_interval
                for the online training path).

        Returns:
            dict: {"score": float (exp of the log-prob, in [0,1]),
                   "log_prob": float, "token_count": int,
                   "response": str, "success": bool}
        """
        encoded_frames = []
        for frame in frames:
            if frame is None:
                continue
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            encoded_frames.append(self._encode_image(frame))

        if not encoded_frames:
            return {"score": 0.0, "log_prob": None, "token_count": 0,
                    "response": "", "success": False}

        request_data = {
            "frames": encoded_frames,
            "task_description": task_description,
            "fps": fps,
        }

        try:
            response = requests.post(
                f"{self.server_url}/compute_topreward",
                json=request_data,
                timeout=self.timeout,
            )
            result = response.json()
            if not result.get("success", False):
                print(f"VLM topreward failed: {result.get('error', 'Unknown error')}")
                return {"score": 0.0, "log_prob": None, "token_count": 0,
                        "response": "", "success": False}
            return {
                "score": result["score"],
                "log_prob": result.get("log_prob"),
                "token_count": result.get("token_count", 0),
                "response": result.get("response", ""),
                "success": True,
            }
        except requests.exceptions.Timeout:
            print("VLM topreward request timed out")
            return {"score": 0.0, "log_prob": None, "token_count": 0,
                    "response": "", "success": False}
        except Exception as e:
            print(f"VLM topreward request failed: {e}")
            return {"score": 0.0, "log_prob": None, "token_count": 0,
                    "response": "", "success": False}

    def compute_reward(
        self,
        image: np.ndarray,
        task_description: str,
        reward_type: str = "progress",
        goal_image: np.ndarray = None,
        initial_image: np.ndarray = None,
    ) -> dict:
        """
        Compute the VLM reward for a single image.

        Args:
            image: Current observation RGB image, numpy array, shape (H, W, 3), dtype uint8.
            task_description: Task description.
            reward_type: Reward type, "progress" or "quality".
            goal_image: (optional) Goal-state RGB image.
            initial_image: (optional) Initial-state RGB image.

        Returns:
            dict: {"score": float, "response": str, "success": bool}
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        image_bytes = image.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        request_data = {
            "image": image_b64,
            "image_shape": list(image.shape),
            "image_dtype": str(image.dtype),
            "task_description": task_description,
            "reward_type": reward_type,
        }
        
        if goal_image is not None:
            if goal_image.dtype != np.uint8:
                goal_image = (goal_image * 255).astype(np.uint8)
            request_data["goal_image"] = base64.b64encode(goal_image.tobytes()).decode('utf-8')
            request_data["goal_image_shape"] = list(goal_image.shape)
            request_data["goal_image_dtype"] = str(goal_image.dtype)
        
        if initial_image is not None:
            if initial_image.dtype != np.uint8:
                initial_image = (initial_image * 255).astype(np.uint8)
            request_data["initial_image"] = base64.b64encode(initial_image.tobytes()).decode('utf-8')
            request_data["initial_image_shape"] = list(initial_image.shape)
            request_data["initial_image_dtype"] = str(initial_image.dtype)
        
        try:
            response = requests.post(
                f"{self.server_url}/compute_reward",
                json=request_data,
                timeout=self.timeout
            )
            
            result = response.json()
            if not result.get("success", False):
                print(f"VLM reward computation failed: {result.get('error', 'Unknown error')}")
                return {"score": 0.0, "response": "", "success": False}
            
            return {
                "score": result["score"],
                "response": result.get("response", ""),
                "success": True,
            }
        except requests.exceptions.Timeout:
            print("VLM reward request timed out")
            return {"score": 0.0, "response": "", "success": False}
        except Exception as e:
            print(f"VLM reward request failed: {e}")
            return {"score": 0.0, "response": "", "success": False}

    def compute_robometer(
        self,
        frames: list[np.ndarray],
        task_description: str,
    ) -> dict:
        """
        Score a video frame sequence using Robometer/RoboReward-style evaluation.

        Args:
            frames: List of RGB frames, each shape=(H,W,3), dtype=uint8.
            task_description: Task description.

        Returns:
            dict: {"score": float, "raw_score": Optional[int], "response": str, "progress_per_frame": list[float], "success": bool}
        """
        encoded_frames = []
        for frame in frames:
            if frame is None:
                continue
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            encoded_frames.append(self._encode_image(frame))

        if not encoded_frames:
            return {
                "score": 0.0,
                "raw_score": None,
                "response": "",
                "progress_per_frame": [],
                "success": False,
            }

        request_data = {
            "frames": encoded_frames,
            "task_description": task_description,
        }

        try:
            response = requests.post(
                f"{self.server_url}/compute_robometer",
                json=request_data,
                timeout=self.timeout,
            )
            result = response.json()
            if not result.get("success", False):
                print(f"VLM robometer request failed: {result.get('error', 'Unknown error')}")
                return {
                    "score": 0.0,
                    "raw_score": None,
                    "response": "",
                    "progress_per_frame": [],
                    "success": False,
                }
            return {
                "score": result.get("score", 0.0),
                "raw_score": result.get("raw_score"),
                "response": result.get("response", ""),
                "progress_per_frame": result.get("progress_per_frame", []),
                "success": True,
            }
        except requests.exceptions.Timeout:
            print("VLM robometer request timed out")
            return {
                "score": 0.0,
                "raw_score": None,
                "response": "",
                "progress_per_frame": [],
                "success": False,
            }
        except Exception as e:
            print(f"VLM robometer request failed: {e}")
            return {
                "score": 0.0,
                "raw_score": None,
                "response": "",
                "progress_per_frame": [],
                "success": False,
            }
    
    def compute_rewards_batch(
        self,
        images: List[np.ndarray],
        task_descriptions: Union[str, List[str]],
        reward_type: str = "progress",
    ) -> List[dict]:
        """
        Compute VLM rewards for a batch of images.

        Args:
            images: List of images.
            task_descriptions: Task description(s) (a single string or a list).
            reward_type: Reward type.

        Returns:
            List[dict]: Result for each image.
        """
        if isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions] * len(images)
        
        encoded_images = []
        for image in images:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            encoded_images.append(self._encode_image(image))
        
        try:
            response = requests.post(
                f"{self.server_url}/compute_rewards_batch",
                json={
                    "images": encoded_images,
                    "task_descriptions": task_descriptions,
                    "reward_type": reward_type,
                },
                timeout=self.timeout * len(images)  # scale timeout for batch size
            )
            
            result = response.json()
            if not result.get("success", False):
                print(f"VLM reward batch computation failed: {result.get('error', 'Unknown error')}")
                return [{"score": 0.0, "response": "", "success": False}] * len(images)
            
            return [
                {"score": r["score"], "response": r.get("response", ""), "success": True}
                for r in result["results"]
            ]
        except Exception as e:
            print(f"VLM reward batch request failed: {e}")
            return [{"score": 0.0, "response": "", "success": False}] * len(images)


class VLMRewardWrapper:
    """
    VLM Reward Wrapper - wraps VLM reward computation with caching and relative rewards.

    Designed for integration into LiberoEnv.
    """
    
    def __init__(
        self,
        server_url: str = "http://host.docker.internal:5001",
        reward_type: str = "progress",
        reward_scale: float = 5.0,
        use_relative_reward: bool = True,
        sparse_reward_weight: float = 0.3,
        vlm_reward_weight: float = 0.7,
    ):
        """
        Initialize the VLM Reward Wrapper.

        Args:
            server_url: VLM Reward Server address.
            reward_type: Reward type.
            reward_scale: Reward scaling factor.
            use_relative_reward: Whether to use relative reward (current score - previous step score).
            sparse_reward_weight: Weight for sparse environment reward.
            vlm_reward_weight: Weight for VLM reward.
        """
        self.client = VLMRewardClient(server_url)
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.use_relative_reward = use_relative_reward
        self.sparse_reward_weight = sparse_reward_weight
        self.vlm_reward_weight = vlm_reward_weight
        
        # Cache previous-step VLM scores for relative reward computation
        self.prev_vlm_scores = None


        if not self.client.health_check():
            print("WARNING: VLM Reward Server is not available!")
    
    def reset(self, num_envs: int):
        """Reset cached scores (call on environment reset)."""
        self.prev_vlm_scores = np.zeros(num_envs, dtype=np.float32)
    
    def compute_combined_reward(
        self,
        images: List[np.ndarray],
        task_descriptions: Union[str, List[str]],
        sparse_rewards: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """
        Compute combined reward (VLM reward + sparse reward).

        Args:
            images: List of current observation images.
            task_descriptions: Task description(s).
            sparse_rewards: Sparse rewards from the environment.
            dones: Done flags.

        Returns:
            Combined rewards and raw VLM scores.
        """
        num_envs = len(images)
        
        if self.prev_vlm_scores is None or len(self.prev_vlm_scores) != num_envs:
            self.prev_vlm_scores = np.zeros(num_envs, dtype=np.float32)
        
        results = self.client.compute_rewards_batch(images, task_descriptions, self.reward_type)
        vlm_scores = np.array([r["score"] for r in results], dtype=np.float32)
        
        if self.use_relative_reward:
            vlm_rewards = (vlm_scores - self.prev_vlm_scores) * self.reward_scale
        else:
            vlm_rewards = vlm_scores * self.reward_scale
        
        self.prev_vlm_scores = vlm_scores.copy()

        # Reset cached scores for finished environments
        if dones is not None:
            self.prev_vlm_scores[dones] = 0.0
        
        combined_rewards = (
            self.sparse_reward_weight * sparse_rewards +
            self.vlm_reward_weight * vlm_rewards
        )
        
        return combined_rewards, vlm_scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://localhost:5001")
    args = parser.parse_args()
    
    client = VLMRewardClient(server_url=args.server_url)
    
    print("Checking server health...")
    if client.health_check():
        print("Server is healthy!")
    else:
        print("Server is not available!")
        exit(1)
    
    print("\nTesting reward computation...")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    task_description = "pick up the red cube and place it on the blue plate"
    
    result = client.compute_reward(test_image, task_description)
    print(f"Score: {result['score']}")
    print(f"Response: {result['response']}")
    print(f"Success: {result['success']}")
