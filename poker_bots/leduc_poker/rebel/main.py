import torch
import time
from .train import ReBeLTrainer
from .eval import evaluate

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = ReBeLTrainer(device=device)
    
    num_epochs = 10
    games_per_epoch = 10 # Small for demo
    train_steps = 50
    
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 1. Self Play
        trainer.generate_data(num_games=games_per_epoch)
        
        # 2. Train
        loss = trainer.train(batch_size=16, steps=train_steps)
        
        # 3. Eval
        if epoch % 2 == 0:
            avg_payoff = evaluate(trainer.value_net, num_games=20, device=device)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Avg Payoff vs Random: {avg_payoff:.4f} | Time: {time.time()-start_time:.1f}s")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Time: {time.time()-start_time:.1f}s")
            
    print("Training Complete.")
    
    # Final Eval
    final_payoff = evaluate(trainer.value_net, num_games=100, device=device)
    print(f"Final Average Payoff vs Random (100 games): {final_payoff:.4f}")

if __name__ == "__main__":
    main()

