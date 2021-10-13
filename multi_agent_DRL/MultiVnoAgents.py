"""
Some system parameters
"""



class MultiVnoAgents:
    def __init__(self,
                 model,
                 target_model
                 ) -> None:
        """ initialize the Double DQN agent

        Args:
            model: evaluation network
            target_model: target network
            learning_rate: learning rate. Defaults to 0.001.
        """