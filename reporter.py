import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Reporter(SummaryWriter):
    def __init__(self, log_dir: str, snapshot_directory=None):
        self.log_dir = log_dir
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.file_folder = log_dir + '/' + current_time
        super().__init__(self.file_folder)

        if snapshot_directory is None:
            snapshot_directory = self.file_folder + '/snapshots'
        self.snapshot_directory = snapshot_directory
        self.snapshot_count = 0

   # Fix bug where this becomes a recursive copy 
    def copy_to_snapshot(self):
        shutil.copytree(self.file_folder, self.snapshot_directory + '/snapshot_' + str(self.snapshot_count), ignore=shutil.ignore_patterns(self.file_folder))
        self.snapshot_count += 1