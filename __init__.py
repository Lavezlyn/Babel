"""
Resource Exchange Game

Implements the 4-player, 2-team resource exchange game with timesteps per
round. Uses core Channel/VocabularyFilter/LLMAgent utilities.
"""

from config import ResourceExchangeConfig
from pairing import PairingManager
from vocabulary import AlienVocabularyGenerator
from resources import ResourceManager
from scoring import ScoreCalculator
from agent import ResourceExchangeAgent, create_resource_exchange_action_space
from game import ResourceExchangeGame

__all__ = [
    "ResourceExchangeConfig",
    "PairingManager",
    "AlienVocabularyGenerator",
    "ResourceManager",
    "ScoreCalculator",
    "ResourceExchangeAgent",
    "create_resource_exchange_action_space",
    "ResourceExchangeGame",
]

