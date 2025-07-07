import torch
import torch.nn as nn
import torch.optim as optim
from environment import ChessEnv
from model import ChessNet
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

def train():
    # Create directories for saving models and plots
    save_dir = "models"
    plots_dir = "training_plots"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize lists to track metrics
    rewards_history = []
    loss_history = []
    
    stockfish_path = r"H:\Desktop\chessproject\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    print(f"Starting training with Stockfish at: {stockfish_path}")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    env = ChessEnv(stockfish_path)
    
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_episodes = 1000
    start_time = time.time()
    
    print("\nStarting training...")
    print("=" * 50)
    
    for episode in range(num_episodes):
        episode_start = time.time()
        state = env.reset()
        total_reward = 0
        episode_losses = []
        moves = 0
        done = False
        
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
            episode_losses.append(loss.item())
        
        # Track metrics
        avg_loss = sum(episode_losses) / len(episode_losses)
        loss_history.append(avg_loss)
        rewards_history.append(total_reward)
        
        # Calculate time metrics
        episode_time = time.time() - episode_start
        total_time = time.time() - start_time
        
        # Print detailed episode information
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"Moves: {moves}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Episode Time: {episode_time:.2f}s")
        print(f"Total Training Time: {total_time:.2f}s")
        print("-" * 30)
        
        # Save model and plot every 100 episodes
        if (episode + 1) % 100 == 0:
            # Save model
            try:
                save_path = os.path.join(save_dir, f'chess_model_episode_{episode+1}.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
            
            # Plot training progress
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(rewards_history)
            plt.title('Rewards over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            
            plt.subplot(1, 2, 2)
            plt.plot(loss_history)
            plt.title('Loss over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Average Loss')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'training_progress_{episode+1}.png'))
            plt.close()
    
    env.close()
    
    # Final training summary
    print("\nTraining Complete!")
    print("=" * 50)
    print(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Final average reward: {sum(rewards_history[-100:])/100:.2f}")
    print(f"Final average loss: {sum(loss_history[-100:])/100:.4f}")
    print(f"Models saved in: {save_dir}")
    print(f"Training plots saved in: {plots_dir}")

if __name__ == "__main__":
    train() 