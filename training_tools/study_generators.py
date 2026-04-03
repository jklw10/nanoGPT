from collections import deque
import torch
import optuna

class OptunaStudyWrapper:
    def __init__(self, active_space, struct_keys, device):
        self.study = optuna.create_study(direction="maximize")
        self.active_space = active_space
        self.struct_keys = struct_keys
        self.device = device

    def ask(self, batch_size):
        trials =[self.study.ask() for _ in range(batch_size)]
        active_tensors = {
            name: torch.tensor([t.suggest_float(name, sp['low'], sp['high']) for t in trials], device=self.device)
            for name, sp in self.active_space.items()
        }
        return active_tensors, trials  # Return trials as context for the update loop

    def update(self, context, fitnesses, struct_vals):
        trials = context
        for t, f in zip(trials, fitnesses):
            self.study.tell(t, f)

    def get_tensors(self):
        df = self.study.trials_dataframe()
        if df.empty or 'state' not in df.columns: return None
        df = df[df['state'] == 'COMPLETE'].copy()
        if len(df) < 20: return None
            
        top_df = df[df['value'] >= df['value'].quantile(0.90)]
        rename_map = {c: c.replace('params_', '') for c in top_df.columns}
        top_df = top_df.rename(columns=rename_map)
        
        all_cols = list(self.active_space.keys()) + self.struct_keys
        present_cols = [c for c in all_cols if c in top_df.columns]

        try:
            evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
            imp_dict = optuna.importance.get_param_importances(self.study, evaluator=evaluator)
        except Exception: 
            return None
            
        X = torch.tensor(top_df[present_cols].to_numpy(), dtype=torch.float32, device=self.device)
        imp = torch.tensor([imp_dict.get(c, 0.0) if c not in self.struct_keys else 1.0 for c in present_cols], device=self.device)
        struct = torch.tensor([c in self.struct_keys for c in present_cols], dtype=torch.bool, device=self.device)
        
        return X, imp, struct, present_cols


class TensorStudyWrapper:
    def __init__(self, active_space, struct_keys, device, max_history=1000):
        self.active_space = active_space
        self.struct_keys = struct_keys
        self.device = device
        self.history_X = deque(maxlen=max_history)
        self.history_y = deque(maxlen=max_history)
        
    def ask(self, batch_size):
        active_tensors = {
            name: torch.empty(batch_size, device=self.device).uniform_(sp['low'], sp['high'])
            for name, sp in self.active_space.items()
        }
        return active_tensors, active_tensors 
    
    def update(self, context, fitnesses, struct_vals):
        active_tensors = context
        batch_size = len(fitnesses)
        
        X_active = torch.stack([active_tensors[k] for k in self.active_space.keys()], dim=1) if self.active_space else torch.empty((batch_size, 0), device=self.device)
        struct_tensor = torch.tensor(struct_vals, dtype=torch.float32, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        self.history_X.append(torch.cat([X_active, struct_tensor], dim=1))
        self.history_y.append(torch.tensor(fitnesses, dtype=torch.float32, device=self.device))

    def get_tensors(self):
        if sum(y.shape[0] for y in self.history_y) < 20: return None
            
        X_all, y_all = torch.cat(list(self.history_X), dim=0), torch.cat(list(self.history_y), dim=0).squeeze()
        
        kth = max(1, int(y_all.shape[0] * 0.90))
        mask = y_all >= torch.kthvalue(y_all, y_all.shape[0] - kth + 1).values
        X_top, y_top = X_all[mask], y_all[mask].unsqueeze(-1)
        
        present_cols = list(self.active_space.keys()) + self.struct_keys
        
        X_c, y_c = X_top - X_top.mean(dim=0, keepdim=True), y_top - y_top.mean(dim=0, keepdim=True)
        std_x, std_y = X_c.std(dim=0) * (X_top.shape[0] - 1)**0.5, y_c.std(dim=0) * (y_top.shape[0] - 1)**0.5
        corr = (X_c * y_c).sum(dim=0) / (std_x * std_y.squeeze() + 1e-8)
        
        imp = torch.tensor([corr.abs()[i] if c not in self.struct_keys else 1.0 for i, c in enumerate(present_cols)], device=self.device)
        struct = torch.tensor([c in self.struct_keys for c in present_cols], dtype=torch.bool, device=self.device)
        
        return X_top, imp, struct, present_cols