import torch
import pandas

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
                )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class Logger():
    # Author: Jacob Schreiber <jmschreiber91@gmail.com>
	"""A logging class that can report or save metrics.

	This class contains a simple utility for saving statistics as they are
	generated, saving a report to a text file at the end, and optionally
	print the report to screen one line at a time as it is being generated.
	Must begin using the `start` method, which will reset the logger.
	
	
	Parameters
	----------
	names: list or tuple
		An iterable containing the names of the columns to be logged.
	
	verbose: bool, optional
		Whether to print to screen during the logging.
	"""

	def __init__(self, names, verbose=False):
		self.names = names
		self.verbose = verbose

	def start(self):
		"""Begin the recording process."""

		self.data = {name: [] for name in self.names}

		if self.verbose:
			print("\t".join(self.names))

	def add(self, row):
		"""Add a row to the log.

		This method will add one row to the log and, if verbosity is set,
		will print out the row to the log. The row must be the same length
		as the names given at instantiation.
		

		Parameters
		----------
		args: tuple or list
			An iterable containing the statistics to be saved.
		"""

		assert len(row) == len(self.names)

		for name, value in zip(self.names, row):
			self.data[name].append(value)

		if self.verbose:
			print("\t".join(map(str, [round(x, 4) if isinstance(x, float) else x 
				for x in row])))

	def save(self, name):
		"""Write a log to disk.

		
		Parameters
		----------
		name: str
			The filename to save the logs to.
		"""

		pandas.DataFrame(self.data).to_csv(name, sep='\t', index=False)