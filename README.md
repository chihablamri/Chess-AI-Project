# Chess AI Project

## Description
A Python-based Chess AI project that allows users to train, evaluate, and play against a neural network chess engine. The project features:
- A web interface to play chess against the AI (with drag-and-drop or text input)
- Training scripts to train the AI using reinforcement learning and Stockfish
- Tools to continue training, train from game data, and play against the AI in the terminal
- Visualization of training progress

## Features
- Play chess against a neural network AI in your browser
- Train the AI using Stockfish as an opponent
- Continue training from saved models
- Train from a dataset of chess games (CSV)
- Visualize training progress with plots

## Installation
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Stockfish:**
   - Download the Stockfish chess engine from https://stockfishchess.org/download/
   - Place the executable in a known location (see code for expected paths or update the path in the scripts)

## Usage
### 1. Train the AI
- Run a full training session:
  ```bash
  python src/train.py
  ```
- Quick training (10 episodes):
  ```bash
  python src/quick_train.py
  ```
- Continue training from a saved model:
  ```bash
  python src/continue_training.py
  ```
- Train from a CSV of games:
  ```bash
  python src/train_from_games.py
  ```

### 2. Play Against the AI
- **Web App:**
  ```bash
  python src/chess_web_app.py
  # or
  python src/chess_visual_app.py
  # or
  python src/simple_chess_app.py
  ```
  Then open your browser to the indicated address (usually http://127.0.0.1:5000 or similar)

- **Terminal:**
  ```bash
  python src/play_against_ai.py
  ```

### 3. View Training Progress
- Training plots are saved in the `training_plots/` directory.

## Requirements
- Python 3.7+
- See `requirements.txt` for Python dependencies
- Stockfish chess engine (external download)

## Credits
- Uses [python-chess](https://python-chess.readthedocs.io/), [Flask](https://flask.palletsprojects.com/), [PyTorch](https://pytorch.org/), and [Stockfish](https://stockfishchess.org/)

## License
MIT License (or specify your license here) 