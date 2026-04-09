#!/usr/bin/env python3
"""
VLM Reward Client - 在 Docker 容器中使用，通过 HTTP 请求获取 VLM reward

用法：
    from vlm_reward_client import VLMRewardClient
    
    client = VLMRewardClient(server_url="http://host.docker.internal:5001")
    reward = client.compute_reward(image, task_description)
"""

import base64
from typing import List, Optional, Union

import numpy as np
import requests


class VLMRewardClient:
    """VLM Reward Client - 通过 HTTP 与 VLM Reward Server 通信"""
    
    def __init__(
        self,
        server_url: str = "http://host.docker.internal:5001",
        timeout: float = 30.0,
    ):
        """
        初始化 VLM Reward Client
        
        Args:
            server_url: VLM Reward Server 的地址
                - 如果在 Docker 中访问宿主机：http://host.docker.internal:5001
                - 如果在同一机器：http://localhost:5001
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
    def health_check(self) -> bool:
        """检查服务器是否可用"""
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
        """将图像编码为 base64"""
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
        判断任务是否完成 (yes/no)

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
    ) -> dict:
        """
        Score a trajectory via RoboReward (1-5 discrete, normalized to 0-1).

        Args:
            frames: RGB frame list, each shape=(H,W,3), dtype=uint8
            task_description: Task description

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

    def compute_reward(
        self,
        image: np.ndarray,
        task_description: str,
        reward_type: str = "progress",
        goal_image: np.ndarray = None,
        initial_image: np.ndarray = None,
    ) -> dict:
        """
        计算单张图像的 VLM reward
        
        Args:
            image: 当前观察的 RGB 图像，numpy array，shape (H, W, 3)，dtype uint8
            task_description: 任务描述
            reward_type: 奖励类型，"progress" 或 "quality"
            goal_image: （可选）目标状态的 RGB 图像
            initial_image: （可选）初始状态的 RGB 图像
            
        Returns:
            dict: {"score": float, "response": str, "success": bool}
        """
        # 确保图像格式正确
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 编码图像
        image_bytes = image.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 构建请求数据
        request_data = {
            "image": image_b64,
            "image_shape": list(image.shape),
            "image_dtype": str(image.dtype),
            "task_description": task_description,
            "reward_type": reward_type,
        }
        
        # 添加可选的 goal image
        if goal_image is not None:
            if goal_image.dtype != np.uint8:
                goal_image = (goal_image * 255).astype(np.uint8)
            request_data["goal_image"] = base64.b64encode(goal_image.tobytes()).decode('utf-8')
            request_data["goal_image_shape"] = list(goal_image.shape)
            request_data["goal_image_dtype"] = str(goal_image.dtype)
        
        # 添加可选的 initial image
        if initial_image is not None:
            if initial_image.dtype != np.uint8:
                initial_image = (initial_image * 255).astype(np.uint8)
            request_data["initial_image"] = base64.b64encode(initial_image.tobytes()).decode('utf-8')
            request_data["initial_image_shape"] = list(initial_image.shape)
            request_data["initial_image_dtype"] = str(initial_image.dtype)
        
        # 发送请求
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
        使用视频帧序列进行 Robometer/RoboReward 风格打分。

        Args:
            frames: RGB 帧列表，元素 shape=(H,W,3), dtype=uint8
            task_description: 任务描述

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
        批量计算 VLM rewards
        
        Args:
            images: 图像列表
            task_descriptions: 任务描述（单个字符串或列表）
            reward_type: 奖励类型
            
        Returns:
            List[dict]: 每张图像的结果
        """
        if isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions] * len(images)
        
        # 编码所有图像
        encoded_images = []
        for image in images:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            encoded_images.append(self._encode_image(image))
        
        # 发送请求
        try:
            response = requests.post(
                f"{self.server_url}/compute_rewards_batch",
                json={
                    "images": encoded_images,
                    "task_descriptions": task_descriptions,
                    "reward_type": reward_type,
                },
                timeout=self.timeout * len(images)  # 批量请求增加超时
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
    VLM Reward Wrapper - 封装 VLM reward 计算，支持缓存和相对奖励
    
    用于集成到 LiberoEnv 中
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
        初始化 VLM Reward Wrapper
        
        Args:
            server_url: VLM Reward Server 地址
            reward_type: 奖励类型
            reward_scale: 奖励缩放系数
            use_relative_reward: 是否使用相对奖励（当前分数 - 上一步分数）
            sparse_reward_weight: 稀疏奖励权重
            vlm_reward_weight: VLM 奖励权重
        """
        self.client = VLMRewardClient(server_url)
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.use_relative_reward = use_relative_reward
        self.sparse_reward_weight = sparse_reward_weight
        self.vlm_reward_weight = vlm_reward_weight
        
        # 缓存上一步的 VLM scores（用于计算相对奖励）
        self.prev_vlm_scores = None
        
        # 检查服务器是否可用
        if not self.client.health_check():
            print("WARNING: VLM Reward Server is not available!")
    
    def reset(self, num_envs: int):
        """重置缓存（在环境重置时调用）"""
        self.prev_vlm_scores = np.zeros(num_envs, dtype=np.float32)
    
    def compute_combined_reward(
        self,
        images: List[np.ndarray],
        task_descriptions: Union[str, List[str]],
        sparse_rewards: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """
        计算组合奖励（VLM reward + sparse reward）
        
        Args:
            images: 当前观测图像列表
            task_descriptions: 任务描述
            sparse_rewards: 环境的稀疏奖励
            dones: 是否结束
            
        Returns:
            组合奖励
        """
        num_envs = len(images)
        
        # 初始化 prev_vlm_scores
        if self.prev_vlm_scores is None or len(self.prev_vlm_scores) != num_envs:
            self.prev_vlm_scores = np.zeros(num_envs, dtype=np.float32)
        
        # 计算 VLM rewards
        results = self.client.compute_rewards_batch(images, task_descriptions, self.reward_type)
        vlm_scores = np.array([r["score"] for r in results], dtype=np.float32)
        
        # 计算 VLM reward
        if self.use_relative_reward:
            vlm_rewards = (vlm_scores - self.prev_vlm_scores) * self.reward_scale
        else:
            vlm_rewards = vlm_scores * self.reward_scale
        
        # 更新缓存
        self.prev_vlm_scores = vlm_scores.copy()
        
        # 对于结束的环境，重置缓存
        if dones is not None:
            self.prev_vlm_scores[dones] = 0.0
        
        # 组合奖励
        combined_rewards = (
            self.sparse_reward_weight * sparse_rewards +
            self.vlm_reward_weight * vlm_rewards
        )
        
        return combined_rewards, vlm_scores


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://localhost:5001")
    args = parser.parse_args()
    
    client = VLMRewardClient(server_url=args.server_url)
    
    # 健康检查
    print("Checking server health...")
    if client.health_check():
        print("Server is healthy!")
    else:
        print("Server is not available!")
        exit(1)
    
    # 测试计算奖励
    print("\nTesting reward computation...")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    task_description = "pick up the red cube and place it on the blue plate"
    
    result = client.compute_reward(test_image, task_description)
    print(f"Score: {result['score']}")
    print(f"Response: {result['response']}")
    print(f"Success: {result['success']}")
