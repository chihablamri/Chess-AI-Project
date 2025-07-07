import gym
import chess
import chess.engine
import numpy as np
from typing import Tuple, Dict, Any

class ChessEnv(gym.Env):
    """Custom Chess RL environment."""
    
    def __init__(self, stockfish_path: str):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(list(self.board.legal_moves)))
        # 8x8x12 for pieces (6 piece types x 2 colors) + 1 channel for turn
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(8, 8, 13), dtype=np.float32
        )
        
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.previous_positions = set()

    def get_state(self) -> np.ndarray:
        """Converts board state into a 8x8x13 numpy array."""
        state = np.zeros((8, 8, 13), dtype=np.float32)
        
        # Fill piece planes
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                rank, file = i // 8, i % 8
                # Get piece plane index (0-11)
                piece_idx = (piece.piece_type - 1) + (6 if piece.color else 0)
                state[rank, file, piece_idx] = 1
                
        # Fill turn plane
        state[:, :, 12] = float(self.board.turn)
        
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return new state, reward, done, info."""
        legal_moves = list(self.board.legal_moves)
        if action >= len(legal_moves):
            return self.get_state(), -10, True, {'error': 'Invalid move'}
            
        move = legal_moves[action]
        self.board.push(move)
        
        # Get state before checking game over
        new_state = self.get_state()
        
        # Check game over conditions
        if self.board.is_game_over():
            reward = self._get_game_over_reward()
            return new_state, reward, True, {'outcome': self.board.outcome()}
            
        # Calculate reward based on position evaluation and novelty
        reward = self._calculate_reward()
        
        return new_state, reward, False, {}

    def _calculate_reward(self) -> float:
        """Calculate reward based on position evaluation and novelty."""
        # Get position evaluation from engine
        info = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
        eval_score = info['score'].relative.score(mate_score=10000)
        
        # Novelty bonus
        pos_hash = hash(self.board.fen())
        novelty_bonus = 0.5 if pos_hash not in self.previous_positions else 0
        self.previous_positions.add(pos_hash)
        
        return float(eval_score) / 100.0 + novelty_bonus

    def _get_game_over_reward(self) -> float:
        """Calculate reward for game over position."""
        outcome = self.board.outcome()
        if outcome.winner == self.board.turn:
            return 10.0  # Win
        elif outcome.winner is None:
            return 0.0   # Draw
        else:
            return -10.0 # Loss

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.board.reset()
        self.previous_positions.clear()
        return self.get_state()

    def close(self):
        """Clean up resources."""
        if self.engine:
            self.engine.quit() 