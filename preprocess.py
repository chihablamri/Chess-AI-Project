import pandas as pd
import chess
import chess.pgn
import io

def load_data(file_path):
    """Loads chess game data from a CSV file."""
    return pd.read_csv(file_path)

def extract_positions(pgn_string):
    """Extracts FEN positions from PGN game data."""
    # Create a string IO object for chess.pgn to read
    pgn = chess.pgn.read_game(io.StringIO(pgn_string))
    if not pgn:
        return []
        
    positions = []
    board = pgn.board()
    
    for move in pgn.mainline_moves():
        board.push(move)
        positions.append({
            'fen': board.fen(),
            'turn': board.turn,
            'legal_moves': [str(move) for move in board.legal_moves]
        })
    
    return positions

def save_processed_data(positions, output_file):
    """Saves extracted positions data to a new file."""
    df = pd.DataFrame(positions)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    data = load_data("data/games.csv")
    all_positions = []

    for pgn in data["PGN"]:
        all_positions.extend(extract_positions(pgn))

    save_processed_data(all_positions, "data/positions.csv") 