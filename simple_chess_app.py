from flask import Flask, render_template, request, jsonify
import chess
import torch
import os
import numpy as np
from model import ChessNet

app = Flask(__name__)

# Global variables
board = chess.Board()
player_color = chess.WHITE
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    
    # Find the model file
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return False
        
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return False
        
    # Try to find a valid model
    try:
        # First try models with episode numbers
        episode_models = [f for f in model_files if f.split('_')[-1].split('.')[0].isdigit()]
        if episode_models:
            episode_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = os.path.join(model_dir, episode_models[-1])
        else:
            # Otherwise use any model
            model_path = os.path.join(model_dir, model_files[0])
            
        # Load the model
        model = ChessNet().to(device)
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            # Try alternative loading method
            model.load_state_dict(torch.load(model_path, map_location=device))
            
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

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

def get_ai_move():
    """Get a move from the AI model."""
    global board, model
    
    try:
        # Check if model is loaded
        if model is None:
            print("Model is None, using random moves")
            # Use random move if model not loaded
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = np.random.choice(legal_moves)
                return move
            return None
            
        # Convert board to state
        state = board_to_state(board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        
        # Get action probabilities from model
        with torch.no_grad():
            action_probs = model(state_tensor)
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("No legal moves available")
            return None
            
        print(f"Number of legal moves: {len(legal_moves)}")
        
        # Filter action probabilities to only legal moves
        legal_move_probs = []
        for i, move in enumerate(legal_moves):
            # Get index in action_probs
            move_idx = i  # This is a simplification
            if move_idx < action_probs.size(1):
                legal_move_probs.append((move, action_probs[0, move_idx].item()))
        
        # If no legal moves with valid probabilities, choose randomly
        if not legal_move_probs:
            print("No legal moves with valid probabilities, choosing randomly")
            return np.random.choice(legal_moves)
        
        # Sort by probability (highest first)
        legal_move_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Return the move with highest probability
        best_move = legal_move_probs[0][0]
        print(f"AI chose move: {best_move}")
        return best_move
        
    except Exception as e:
        print(f"Error in get_ai_move: {e}")
        # Use random move as fallback
        try:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return np.random.choice(legal_moves)
        except:
            pass
        return None

def get_game_status():
    """Get the current game status."""
    global board
    
    if board.is_checkmate():
        return "Checkmate! " + ("You win!" if board.turn != player_color else "AI wins!")
    elif board.is_stalemate():
        return "Game ended in stalemate!"
    elif board.is_insufficient_material():
        return "Game ended due to insufficient material!"
    elif board.is_fifty_moves():
        return "Game ended (fifty-move rule)!"
    elif board.is_repetition():
        return "Game ended (threefold repetition)!"
    elif board.is_check():
        return "Check!"
    else:
        return "Your turn" if board.turn == player_color else "AI's turn"

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Simple Chess AI</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .board {
            font-family: monospace;
            font-size: 18px;
            white-space: pre;
            margin: 20px 0;
            padding: 10px;
            background-color: #f8f8f8;
            border-radius: 5px;
        }
        .controls {
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        button {
            margin: 5px;
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            width: 100%;
            text-align: center;
            font-weight: bold;
        }
        .move-input {
            margin: 15px 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .move-input input {
            padding: 8px;
            margin-right: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            width: 100px;
        }
        .move-input button {
            padding: 8px 15px;
        }
        .move-history {
            margin-top: 20px;
            width: 100%;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .legal-moves {
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Simple Chess AI</h1>
        <div class="controls">
            <div>
                <button id="playAsWhite">Play as White</button>
                <button id="playAsBlack">Play as Black</button>
                <button id="resetGame">Reset Game</button>
            </div>
        </div>
        <div class="status" id="status">Choose a color to start playing</div>
        
        <div class="board" id="board"></div>
        
        <div class="legal-moves" id="legalMoves"></div>
        
        <div class="move-input">
            <input type="text" id="moveInput" placeholder="e.g., e2e4" />
            <button id="submitMove">Make Move</button>
        </div>
        
        <div class="move-history" id="moveHistory">
            <h3>Move History</h3>
            <div id="moves"></div>
        </div>
    </div>

    <script>
        let gameStarted = false;
        let playerColor = 'w';
        
        function updateBoard() {
            $.get('/get_board', function(data) {
                $('#board').text(data.board);
                $('#status').text(data.status);
                $('#legalMoves').text('Legal moves: ' + data.legal_moves.join(', '));
                
                // Update move history
                let html = '';
                for (let i = 0; i < data.move_history.length; i += 2) {
                    const moveNum = Math.floor(i/2) + 1;
                    const whiteMove = data.move_history[i];
                    const blackMove = i+1 < data.move_history.length ? data.move_history[i+1] : '';
                    html += `<div>${moveNum}. ${whiteMove} ${blackMove}</div>`;
                }
                $('#moves').html(html);
            });
        }
        
        function submitMove() {
            if (!gameStarted) {
                alert("Please start a game first!");
                return;
            }
            
            const moveText = $('#moveInput').val().trim();
            if (moveText.length < 4) {
                alert("Please enter a valid move (e.g., e2e4)");
                return;
            }
            
            // Extract from and to squares
            const from = moveText.substring(0, 2);
            const to = moveText.substring(2, 4);
            
            // Make the move
            $.ajax({
                url: '/make_move',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    from: from,
                    to: to
                }),
                success: function(data) {
                    if (data.valid) {
                        // Update the board
                        updateBoard();
                        
                        // Clear the input field
                        $('#moveInput').val('');
                        
                        // If game not over and AI's turn, get AI move
                        if (!data.gameOver) {
                            $('#status').text('AI is thinking...');
                            setTimeout(function() {
                                $.get('/get_ai_move', function(aiData) {
                                    updateBoard();
                                });
                            }, 500);
                        }
                    } else {
                        alert("Invalid move! Please try again. Error: " + data.error);
                    }
                }
            });
        }
        
        function startGame(color) {
            playerColor = color;
            gameStarted = true;
            
            // Reset the game on the server
            $.ajax({
                url: '/reset_game',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    playerColor: color
                }),
                success: function(data) {
                    updateBoard();
                    
                    // If AI goes first (player is black)
                    if (color === 'b') {
                        $('#status').text('AI is thinking...');
                        setTimeout(function() {
                            $.get('/get_ai_move', function(data) {
                                updateBoard();
                            });
                        }, 500);
                    }
                }
            });
        }
        
        $(document).ready(function() {
            // Initial board update
            updateBoard();
            
            // Set up event handlers
            $('#playAsWhite').on('click', function() {
                startGame('w');
            });
            
            $('#playAsBlack').on('click', function() {
                startGame('b');
            });
            
            $('#resetGame').on('click', function() {
                startGame(playerColor);
            });
            
            // Set up text move input handlers
            $('#submitMove').on('click', submitMove);
            $('#moveInput').on('keypress', function(e) {
                if (e.which === 13) { // Enter key
                    submitMove();
                }
            });
        });
    </script>
</body>
</html>
'''

@app.route('/get_board')
def get_board():
    global board
    
    # Get ASCII representation of the board
    board_str = str(board)
    
    # Get legal moves in UCI format
    legal_moves = [move.uci() for move in board.legal_moves]
    
    # Get move history
    move_history = [move.uci() for move in board.move_stack]
    
    return jsonify({
        'board': board_str,
        'status': get_game_status(),
        'legal_moves': legal_moves,
        'move_history': move_history,
        'gameOver': board.is_game_over()
    })

@app.route('/reset_game', methods=['POST'])
def reset_game():
    global board, player_color
    
    data = request.json
    player_color = chess.WHITE if data['playerColor'] == 'w' else chess.BLACK
    
    board = chess.Board()
    
    return jsonify({
        'status': "Your turn" if board.turn == player_color else "AI's turn"
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    global board
    
    data = request.json
    
    try:
        from_square = chess.parse_square(data['from'])
        to_square = chess.parse_square(data['to'])
        
        # Create the move
        move = chess.Move(from_square, to_square)
        
        # Check for promotion
        if data.get('promotion'):
            promotion_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
            promotion = promotion_map.get(data['promotion'].lower())
            if promotion:
                move = chess.Move(from_square, to_square, promotion=promotion)
        
        # Check if move is legal
        if move in board.legal_moves:
            # Make the move
            board.push(move)
            
            # Check if game is over
            game_over = board.is_game_over()
            
            return jsonify({
                'valid': True,
                'gameOver': game_over
            })
        else:
            # List all legal moves for debugging
            legal_moves = [m.uci() for m in board.legal_moves]
            return jsonify({
                'valid': False,
                'error': f"Move {data['from']}{data['to']} is not legal. Legal moves: {', '.join(legal_moves)}"
            })
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        })

@app.route('/get_ai_move')
def ai_move():
    global board
    
    # Get AI move
    move = get_ai_move()
    
    if move:
        # Make the move
        board.push(move)
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'No valid move found'})

if __name__ == '__main__':
    # Load the model
    if load_model():
        print("Model loaded successfully!")
    else:
        print("No model found or error loading model. Please train a model first.")
        print("Running with random moves for demonstration.")
    
    # Run the app
    app.run(debug=True, port=5001) 