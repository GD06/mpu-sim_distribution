from simulator.bank import Bank 


class ProcessingEngine:

    def __init__(self, pe_id, env, config, log, pg):
        self.pe_id = pe_id
        self.env = env
        self.config = config 
        self.log = log 
        self.pg = pg 

        self.bank = Bank(env=env, config=config, log=log, pe=self) 
        return 

    def reset_status(self):
        self.bank.reset_status()
        return 
