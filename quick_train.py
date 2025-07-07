import torch
import torch.nn as nn
import torch.optim as optim
from environment import ChessEnv
from model import ChessNet
import os

def quick_train():
    # Create directory for saving models
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Improved Stockfish path detection
    stockfish_path = None
    possible_paths = [
        r"stockfish\stockfish-windows-x86-64-avx2.exe",
        r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe",
        # Add local directory check
        r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            stockfish_path = path
            break
    
    if stockfish_path is None:
        print("Stockfish not found. Please specify the path to Stockfish.")
        return
    
    print(f"Starting quick training with Stockfish at: {stockfish_path}")
    
    env = ChessEnv(stockfish_path)
    
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for just 10 episodes to get a basic model quickly
    num_episodes = 10
    
    print("\nStarting quick training...")
    print("=" * 50)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        moves = 0
        
        while not done:
            moves += 1
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            state_tensor = state_tensor.permute(0, 3, 1, 2)
            
            # Get action probabilities from model
            action_probs = model(state_tensor)
            
            # Select action
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
            
            # Limit moves per episode to speed up training
            if moves > 50:
                done = True
        
        print(f"Episode {episode + 1}/{num_episodes}, Moves: {moves}, Total Reward: {total_reward:.2f}")
    
    # Save the final model
    save_path = os.path.join(save_dir, 'chess_model_quick_train.pth')
    try:
        torch.save({
            'episode': num_episodes,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"\nModel saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    env.close()
    print("\nQuick training complete! You can now play against the AI.")

if __name__ == "__main__":
    quick_train() 