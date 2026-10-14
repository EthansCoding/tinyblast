import math
import game
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class tinyBlasterEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Movement: 3 choices, rotation: 3 choices, shooting: 2.
        self.action_space = spaces.MultiDiscrete([3, 3, 2])

        # Each player's x/y and heading cos/sin, then relative enemy bearing cos/sin.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Create your simulation here.
        self.handler = game.GameHandler()
        self.MLPlayer = handler.create_player()
        self.SCRIPTPlayer = handler.create_player()


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.handler = game.GameHandler()
        self.MLPlayer = handler.create_player()
        self.SCRIPTPlayer = handler.create_player()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        if not self.stepCount: 
            self.stepCount = 0
        else:
            self.stepCount += 1

        movement = int(action[0]) - 1
        rotation = int(action[1]) - 1
        shoot = bool(action[2])

        self.handler.set_input(
            self.MLPlayer,
            movement,
            rotation,
            shoot
        )
        # Turn along the shortest path toward the ML player.
        offset = self.MLPlayer.position - self.SCRIPTPlayer.position
        target_angle = math.degrees(math.atan2(offset.y, offset.x))
        angle_error = (target_angle - self.SCRIPTPlayer.angle + 180) % 360 - 180

        # A small aiming tolerance prevents jitter around the target angle.
        bot_rotation = 0
        if angle_error > 3:
            bot_rotation = 1
        elif angle_error < -3:
            bot_rotation = -1

        self.handler.set_input(
            self.SCRIPTPlayer,
            movement=0,
            rotation=bot_rotation,
            fire=offset.length_squared() > 0 and abs(angle_error) <= 5,
        )
        self.handler.update()
        
        reward = 0.0
        terminated = False
        truncated = False

        if len(self.handler.players) < 2:
            terminated = True
            if self.handler.players[0] == self.MLPlayer:
                reward = 1.0
            else:
                reward = -1.0

        if self.stepCount > 18000:
            truncated = True   # True when the match hits a time limit.

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        own_angle = math.radians(self.MLPlayer.angle)
        enemy_angle = math.radians(self.SCRIPTPlayer.angle)
        offset = self.SCRIPTPlayer.position - self.MLPlayer.position

        # Express the enemy's bearing relative to the ML player's facing direction.
        # Coincident positions have no bearing, so use zero for both features.
        angle_to_enemy_cos = 0.0
        angle_to_enemy_sin = 0.0
        if offset.length_squared() > 0:
            angle_to_enemy = math.atan2(offset.y, offset.x) - own_angle
            angle_to_enemy_cos = math.cos(angle_to_enemy)
            angle_to_enemy_sin = math.sin(angle_to_enemy)

        return np.array([
            self.MLPlayer.position.x / game.WIDTH,
            self.MLPlayer.position.y / game.HEIGHT,
            math.cos(own_angle),
            math.sin(own_angle),
            self.SCRIPTPlayer.position.x / game.WIDTH,
            self.SCRIPTPlayer.position.y / game.HEIGHT,
            math.cos(enemy_angle),
            math.sin(enemy_angle),
            angle_to_enemy_cos,
            angle_to_enemy_sin,
        ], dtype=np.float32)
