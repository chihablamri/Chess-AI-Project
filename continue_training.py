import torch
import torch.nn as nn
import torch.optim as optim
from training_environment import ChessTrainingEnv
from model import ChessNet
import os
import time

def continue_training(episodes=100, stockfish_path=None):
    """Continue training an existing model for additional episodes."""
    # Create directory for saving models
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    if stockfish_path is None:
        # Try to find Stockfish in common locations
        possible_paths = [
            r"stockfish\stockfish-windows-x86-64-avx2.exe",
            r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe",
            r"H:\Desktop\chessproject\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                stockfish_path = path
                break
        
        if stockfish_path is None:
            print("Stockfish not found. Please specify the path to Stockfish.")
            return
    
    print(f"Starting continued training with Stockfish at: {stockfish_path}")
    
    # Use the training environment instead of the regular environment
    env = ChessTrainingEnv(stockfish_path)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load existing model
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
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
    
    # Train for specified number of episodes
    print(f"\nStarting training for {episodes} more episodes...")
    print("=" * 50)
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        moves = 0
        
        # Play until the game naturally ends (checkmate, stalemate, etc.)
        # But add a safety limit to prevent infinite games (e.g., 500 moves)
        max_moves = 500
        
        while not done and moves < max_moves:
            moves += 1
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            state_tensor = state_tensor.permute(0, 3, 1, 2)
            
            # Get action probabilities from model
            action_probs = model(state_tensor)
            
            # Select action - with some exploration
            if torch.rand(1).item() < 0.1:  # 10% chance of random move for exploration
                action = torch.randint(0, action_probs.size(1), (1,)).item()
            else:
                action = torch.argmax(action_probs).item()
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Calculate loss and update model
            target = torch.zeros_like(action_probs)
            target[0, action] = reward
            loss = criterion(action_probs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
        
        # Record reason for game ending
        if moves >= max_moves:
            end_reason = "move limit reached"
        elif info.get('checkmate', False):
            end_reason = "checkmate"
        elif info.get('stalemate', False):
            end_reason = "stalemate"
        elif info.get('insufficient_material', False):
            end_reason = "insufficient material"
        elif info.get('repetition', False):
            end_reason = "repetition"
        elif info.get('fifty_moves', False):
            end_reason = "fifty-move rule"
        else:
            end_reason = "unknown"
        
        current_episode = start_episode + episode + 1
        print(f"Episode {episode + 1}/{episodes} (Total: {current_episode}), Moves: {moves}, Reward: {total_reward:.2f}, End: {end_reason}")
        
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
    
    env.close()
    return model

if __name__ == "__main__":
    continue_training(episodes=100) 