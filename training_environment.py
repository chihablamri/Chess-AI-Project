import chess
import numpy as np
import random
from stockfish import Stockfish

class ChessTrainingEnv:
    """Chess environment for self-play training."""
    
    def __init__(self, stockfish_path=None):
        self.board = chess.Board()
        self.stockfish = None
        if stockfish_path:
            try:
                self.stockfish = Stockfish(path=stockfish_path)
                print("Stockfish initialized successfully")
            except Exception as e:
                print(f"Error initializing Stockfish: {e}")
                print("Training will proceed without Stockfish evaluation")
    
    def reset(self):
        """Reset the environment to start a new game."""
        self.board = chess.Board()
        return self._get_state()
    
    def _get_state(self):
        """Convert the board to a state representation."""
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
    
    def _get_reward(self):
        """Calculate reward based on board state."""
        # Check if game is over
        if self.board.is_checkmate():
            # Big reward/penalty for checkmate
            return 100.0 if not self.board.turn else -100.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_repetition() or self.board.is_fifty_moves():
            # Small penalty for draw
            return -10.0
            
        # Use Stockfish evaluation if available
        if self.stockfish:
            try:
                self.stockfish.set_fen_position(self.board.fen())
                eval = self.stockfish.get_evaluation()
                
                # Convert evaluation to reward
                if eval['type'] == 'cp':
                    # Centipawn evaluation
                    reward = eval['value'] / 100.0
                    # Negate if black's turn
                    if not self.board.turn:
                        reward = -reward
                    return reward
                elif eval['type'] == 'mate':
                    # Checkmate in X moves
                    mate_in = eval['value']
                    if mate_in > 0:
                        return 50.0  # Positive for winning
                    else:
                        return -50.0  # Negative for losing
            except Exception as e:
                print(f"Stockfish evaluation error: {e}")
        
        # Fallback to material count if Stockfish not available
        white_material = self._count_material(chess.WHITE)
        black_material = self._count_material(chess.BLACK)
        material_diff = white_material - black_material
        
        # Return material difference as reward (positive for white advantage)
        return material_diff if self.board.turn else -material_diff
    
    def _count_material(self, color):
        """Count material value for a given color."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King not counted in material
        }
        
        material = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == color:
                material += values[piece.piece_type]
        
        return material
    
    def step(self, action):
        """Take a step in the environment by making a move."""
        # Get all legal moves
        legal_moves = list(self.board.legal_moves)
        
        # Check if there are legal moves
        if not legal_moves:
            # Game is over
            return self._get_state(), self._get_reward(), True, {
                'checkmate': self.board.is_checkmate(),
                'stalemate': self.board.is_stalemate(),
                'insufficient_material': self.board.is_insufficient_material(),
                'repetition': self.board.is_repetition(),
                'fifty_moves': self.board.is_fifty_moves()
            }
        
        # Make sure action is within range
        if action < 0 or action >= len(legal_moves):
            action = random.randint(0, len(legal_moves) - 1)
        
        # Make the move
        move = legal_moves[action]
        self.board.push(move)
        
        # Check if game is over
        done = self.board.is_game_over()
        
        # Get reward
        reward = self._get_reward()
        
        # Get info
        info = {
            'checkmate': self.board.is_checkmate(),
            'stalemate': self.board.is_stalemate(),
            'insufficient_material': self.board.is_insufficient_material(),
            'repetition': self.board.is_repetition(),
            'fifty_moves': self.board.is_fifty_moves()
        }
        
        # If it's the opponent's turn, make a self-play move
        if not done:
            # Make a random move for the opponent during training
            opponent_moves = list(self.board.legal_moves)
            if opponent_moves:
                opponent_move = random.choice(opponent_moves)
                self.board.push(opponent_move)
                
                # Check if game is over after opponent's move
                done = self.board.is_game_over()
                
                # Update info
                info = {
                    'checkmate': self.board.is_checkmate(),
                    'stalemate': self.board.is_stalemate(),
                    'insufficient_material': self.board.is_insufficient_material(),
                    'repetition': self.board.is_repetition(),
                    'fifty_moves': self.board.is_fifty_moves()
                }
        
        return self._get_state(), reward, done, info
    
    def close(self):
        """Clean up resources."""
        pass 