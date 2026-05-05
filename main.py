import argparse
from engine.trainer import train_model
from engine.evaluator import evaluate_model
from data.mock_data import create_mock_stead_data
import os

def main():
    parser = argparse.ArgumentParser(description="EPGNN - Earthquake Prediction Graph Neural Network")
    parser.add_argument('--mode', type=str, required=True, choices=['mock', 'train', 'evaluate', 'smoke'], 
                        help="Mode to run: mock (generate data), train, evaluate, or smoke (run all three)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size (defaults to 4096 for massive VRAM)")
    parser.add_argument('--model_path', type=str, default='earthquake_gnn.pth', help="Path to saved model weights (.pth)")
    
    # Ablation Flags
    parser.add_argument('--no_cnn', action='store_true', help="Ablate (remove) the Waveform CNN module")
    parser.add_argument('--no_transformer', action='store_true', help="Ablate (remove) the Temporal Transformer module")
    parser.add_argument('--no_gcn', action='store_true', help="Ablate (remove) the GCN spatial module")
    parser.add_argument('--no_dropout', action='store_true', help="Ablate (remove) the Dropout layer")
    
    args = parser.parse_args()
    
    if args.mode in ['mock', 'smoke']:
        print("=== Generating Mock Data ===")
        create_mock_stead_data()
        
    if args.mode in ['train', 'smoke']:
        print("=== Starting Training ===")
        train_model(epochs=args.epochs, batch_size=args.batch_size,
                    use_cnn=not args.no_cnn, use_transformer=not args.no_transformer, use_gcn=not args.no_gcn, use_dropout=not args.no_dropout)
            
    if args.mode in ['evaluate', 'smoke']:
        print("=== Starting Evaluation ===")
        evaluate_model(batch_size=args.batch_size, model_path=args.model_path,
                       use_cnn=not args.no_cnn, use_transformer=not args.no_transformer, use_gcn=not args.no_gcn, use_dropout=not args.no_dropout)

if __name__ == '__main__':
    main()
