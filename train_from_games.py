import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import chess.pgn
import pandas as pd
import io
import os
import time
from model import ChessNet

def board_to_state(board):
    """Convert chess board to state representation."""
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

def train_from_games(csv_file='games.csv', episodes=100, batch_size=32):
    """Train the model from games in a CSV file."""
    # Create directory for saving models
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load existing model or create new one
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate
    criterion = nn.MSELoss()
    
    # Find the latest model
    model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    if not model_files:
        print("No existing model found. Starting with a new model.")
        start_episode = 0
    else:
        # Try to find the latest model
        try:
            # First try models with episode numbers
            episode_models = [f for f in model_files if f.split('_')[-1].split('.')[0].isdigit()]
            if episode_models:
                episode_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_model = os.path.join(save_dir, episode_models[-1])
                start_episode = int(episode_models[-1].split('_')[-1].split('.')[0])
            else:
                # Otherwise use any model
                latest_model = os.path.join(save_dir, model_files[0])
                start_episode = 0
                
            # Load the model
            print(f"Loading model from {latest_model}")
            checkpoint = torch.load(latest_model, map_location=device)
            
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Continuing from episode {start_episode}")
            except:
                # Try alternative loading method
                model.load_state_dict(torch.load(latest_model, map_location=device))
                print("Loaded model weights only (no optimizer state)")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a new model.")
            start_episode = 0
    
    # Load games from CSV
    try:
        print(f"Loading games from {csv_file}...")
        games_df = pd.read_csv(csv_file)
        print(f"Loaded {len(games_df)} games")
        
        # Check what columns we have
        print("CSV columns:", games_df.columns.tolist())
        
        # Determine which column contains the moves
        move_column = None
        for col in ['moves', 'pgn', 'game', 'notation']:
            if col in games_df.columns:
                move_column = col
                break
        
        if move_column is None:
            # Try to find a column that might contain chess moves
            for col in games_df.columns:
                if games_df[col].dtype == 'object' and games_df[col].str.contains('e4|e5|Nf3|d4', case=False).any():
                    move_column = col
                    break
        
        if move_column is None:
            print("Could not find a column with chess moves. Please specify the column name.")
            return
            
        print(f"Using column '{move_column}' for chess moves")
        
        # Check if we have a result column
        result_column = None
        for col in ['result', 'winner', 'outcome']:
            if col in games_df.columns:
                result_column = col
                break
        
        if result_column:
            print(f"Using column '{result_column}' for game results")
        
        # Start training
        print(f"\nStarting training for {episodes} episodes...")
        print("=" * 50)
        
        start_time = time.time()
        
        # Process games in batches
        for episode in range(episodes):
            # Sample a batch of games
            batch_indices = np.random.choice(len(games_df), min(batch_size, len(games_df)), replace=False)
            batch_games = games_df.iloc[batch_indices]
            
            total_positions = 0
            total_loss = 0
            
            for _, game_row in batch_games.iterrows():
                try:
                    # Get the moves
                    moves_text = game_row[move_column]
                    
                    # Skip games with too many moves to avoid memory issues
                    if len(moves_text.split()) > 200:
                        continue
                    
                    # Parse the moves
                    board = chess.Board()
                    moves = moves_text.split()
                    
                    # Process each position in the game
                    positions = []
                    
                    for move_str in moves:
                        try:
                            # Try to parse the move
                            move = board.parse_san(move_str)
                            board.push(move)
                            
                            # Store the position and the move that was played
                            positions.append((board.copy(), move))
                        except ValueError:
                            # Skip invalid moves
                            continue
                    
                    # Train on the positions (limit to 30 positions per game)
                    for board, move in positions[:30]:
                        # Convert board to state
                        state = board_to_state(board)
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        state_tensor = state_tensor.permute(0, 3, 1, 2)
                        
                        # Get all legal moves
                        legal_moves = list(board.legal_moves)
                        if not legal_moves:
                            continue
                        
                        # Create target tensor (one-hot for the move that was played)
                        target = torch.zeros(1, model.fc_policy.out_features).to(device)
                        
                        # Find the index of the move that was played
                        move_idx = 0
                        for i, legal_move in enumerate(legal_moves):
                            if legal_move == move:
                                move_idx = i
                                break
                        
                        # Set the target value
                        if move_idx < model.fc_policy.out_features:
                            target[0, move_idx] = 1.0
                        
                        # Get model prediction
                        output = model(state_tensor)
                        
                        # Ensure output and target have compatible shapes
                        if output.size(1) > target.size(1):
                            output = output[:, :target.size(1)]
                        elif output.size(1) < target.size(1):
                            output = torch.cat([output, torch.zeros(1, target.size(1) - output.size(1)).to(device)], dim=1)
                        
                        # Calculate loss
                        loss = criterion(output, target)
                        
                        # Update model
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_positions += 1
                        total_loss += loss.item()
                
                except Exception as e:
                    # Print error but continue with next game
                    print(f"Error processing game: {e}")
                    continue
            
            # Print progress
            avg_loss = total_loss / max(1, total_positions)
            current_episode = start_episode + episode + 1
            print(f"Episode {episode + 1}/{episodes} (Total: {current_episode}), Positions: {total_positions}, Avg Loss: {avg_loss:.6f}")
            
            # Save model every 10 episodes
            if (episode + 1) % 10 == 0:
                save_path = os.path.join(save_dir, f'chess_model_{current_episode}.pth')
                try:
                    torch.save({
                        'episode': current_episode,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
                    print(f"Model saved to {save_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")
        
        # Save the final model
        final_episode = start_episode + episodes
        save_path = os.path.join(save_dir, f'chess_model_{final_episode}.pth')
        try:
            torch.save({
                'episode': final_episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"\nFinal model saved to {save_path}")
        except Exception as e:
            print(f"Error saving final model: {e}")
        
        training_time = time.time() - start_time
        print(f"\nTraining complete! {episodes} episodes in {training_time:.2f} seconds")
        print(f"Average time per episode: {training_time/episodes:.2f} seconds")
        
        return model
        
    except Exception as e:
        print(f"Error loading or processing games: {e}")
        return None

if __name__ == "__main__":
    train_from_games(episodes=100) 