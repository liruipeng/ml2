import torch
import torch.optim as optim
from typing import List, Callable, Dict
from model.arch import LevelStatus

class Trainer:
    """
    Manages the training loops for multi-level neural networks.
    """
    def __init__(self, 
                 model, 
                 problem, 
                 mesh, 
                 loss_fn: Callable,
                 lr: float = 1e-3,
                 device: str = "cpu"):
        self.model = model
        self.problem = problem
        self.mesh = mesh
        self.loss_fn = loss_fn
        self.lr = lr
        self.device = device

    def train_single_level(self, 
                           level_idx: int, 
                           epochs: int, 
                           n_points: int,
                           callback: Callable = None) -> List[float]:
        """
        Trains a specific level while others are handled based on LevelStatus.
        """
        # Set the target level to TRAIN, ensuring gradients are on
        self.model.set_status(level_idx, LevelStatus.TRAIN)
        
        # Only optimize parameters that require gradients (the active level)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.lr)
        
        history = []
        
        for epoch in range(epochs):
            # Generate new points if using a random sampling approach
            x = self.mesh.generate_interior_points(n_points, method="random")
            
            optimizer.zero_grad()
            loss = self.loss_fn(self.model, self.problem, x)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            history.append(loss_val)
            
            if callback and epoch % 100 == 0:
                callback(epoch, loss_val)
                
        # After training, freeze this level so it contributes but doesn't change
        self.model.set_status(level_idx, LevelStatus.FROZEN)
        return history

    def train_multilevel_sweep(self, 
                               epochs_per_level: int, 
                               n_points: int,
                               sweeps: int = 1):
        """
        Executes the hierarchical training: Level 0 -> Level 1 -> ... Level N.
        """
        for s in range(sweeps):
            print(f"--- Starting Sweep {s+1}/{sweeps} ---")
            for i in range(self.model.num_levels):
                print(f"Training Level {i}...")
                self.train_single_level(i, epochs_per_level, n_points, 
                                        callback=lambda e, l: print(f"  Epoch {e}: Loss {l:.6e}"))

    def evaluate(self, n_test: int = 1000) -> Dict[str, float]:
        """
        Computes L2 and L_inf errors against the exact solution.
        """
        self.model.eval()
        with torch.no_grad():
            x = self.mesh.get_test_grid(n_test)
            u_pred = self.model(x)
            u_true = self.problem.u_exact(x)
            
            error = u_pred - u_true
            l2_error = torch.sqrt(torch.mean(error**2)).item()
            linf_error = torch.max(torch.abs(error)).item()
            
        return {"l2": l2_error, "linf": linf_error}
