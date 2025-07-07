import torch
import chess
import numpy as np
import os
from model import ChessNet
from environment import ChessEnv

def board_to_state(board):
    """Converts chess board to state representation."""
    state = np.zeros((8, 8, 13), dtype=np.float32)
    
    # Fill piece planes
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            rank, file = i // 8, i % 8
            # Get piece plane index (0-11)
            piece_idx = (piece.piece_type - 1) + (6 if piece.color else 0)
            state[rank, file, piece_idx] = 1
            
    # Fill turn plane
    state[:, :, 12] = float(board.turn)
    
    return state

def print_board(board):
    """Print the chess board in a more readable format."""
    print("\n  a b c d e f g h")
    print(" +-----------------+")
    for i in range(8):
        print(f"{8-i}| ", end="")
        for j in range(8):
            piece = board.piece_at(chess.square(j, 7-i))
            if piece:
                print(piece.symbol(), end=" ")
            else:
                print(". ", end="")
        print(f"|{8-i}")
    print(" +-----------------+")
    print("  a b c d e f g h\n")

def get_user_move(board):
    """Get a move from the user."""
    while True:
        try:
            move_str = input("Enter your move (e.g., 'e2e4'): ")
            if move_str.lower() == 'quit':
                return None
                
            # Convert to chess.Move
            move = chess.Move.from_uci(move_str)
            
            # Check if move is legal
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move! Try again.")
        except ValueError:
            print("Invalid input! Please use format like 'e2e4'.")

def get_ai_move(board, model):
    """Get the AI move based on the current board state."""
    # Convert the board to the state representation
    state = board_to_state(board)  # Implement this function to convert the board to the model's input format
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        action_probs = model(state_tensor)
    
    # Select the move with the highest probability (or implement a more sophisticated selection)
    legal_moves = list(board.legal_moves)
    move_probs = action_probs[0].numpy()  # Convert to numpy array
    best_move = max(legal_moves, key=lambda move: move_probs[move])  # Get the best legal move
    
    return best_move

def play_against_ai():
    """Play against the selected AI model."""
    model_name = select_model()
    if not model_name:
        print("No model selected. Exiting.")
        return
    
    # Load the selected model
    model_path = os.path.join("models", model_name)
    checkpoint = torch.load(model_path)
    
    model = ChessNet()
    model.load_state_dict(checkpoint['model_state_dict'])  # Load only the model's state dict
    model.eval()  # Set the model to evaluation mode
    
    board = chess.Board()
    print("Game started! You are playing as White.")
    
    while not board.is_game_over():
        print(board)
        move = input("Enter your move (e.g., 'e2e4') or 'quit' to exit: ")
        if move.lower() == 'quit':
            print("Game aborted.")
            break
        
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move in board.legal_moves:
                board.push(chess_move)
                print(f"You played: {move}")
            else:
                print("Illegal move. Try again.")
                continue
            
            # AI's turn
            ai_move = get_ai_move(board, model)
            board.push(ai_move)
            print(f"AI played: {ai_move.uci()}")
        
        except Exception as e:
            print(f"Error: {e}. Please enter a valid move.")

def list_available_models():
    """List available AI models."""
    model_dir = "models"
    if not os.path.exists(model_dir):
        print("No models found.")
        return []
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        print("No models found.")
        return []
    
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i + 1}: {model_file}")
    
    return model_files

def select_model():
    """Select a model to play against."""
    models = list_available_models()
    if not models:
        return None
    
    choice = input("Select a model by number: ")
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(models):
            return models[choice_index]
        else:
            print("Invalid choice.")
            return None
    except ValueError:
        print("Please enter a valid number.")
        return None

if __name__ == "__main__":
    play_against_ai() 