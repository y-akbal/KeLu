import wandb


class wandb_loss_logger:
    def __init__(self,
                 **cfg):
        wandb.login()
        project_name = cfg["project_name"]
        group = cfg["group_name"]
        name = cfg["group_name"]
        wandb.init(project = project_name,  config = cfg, group = group, name = name)
        
    def log(self, 
            loss:float, 
            log_type = "training_loss"):
        wandb.log(loss)
        
