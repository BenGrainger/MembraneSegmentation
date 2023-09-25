import os
import logging
import json
import datetime 
from pathlib import Path

def get_project_root(root=None):
    if root == None:
        root = Path(__file__).parent

    root_split = os.path.split(root)

    if root_split[1] == "users":

        return root
    else:
        
        return get_project_root(root_split[0])


def check_folder_exists(directory):

    if not os.path.exists(directory):

        try:
            # Create the folder if it doesn't exist
            #os.makedirs(directory)
            print(f"Folder '{directory}' created successfully.")

        except OSError as e:
            print(f"Error creating folder '{directory}': {e}")
    else:
        print(f"Folder '{directory}' already exists.")


class ScriptSetup():
    def __init__(self, config_path, logger=None, out_dir=None):

        self.config_path = config_path
        self.logger = logger
        self.out_dir = out_dir
    
    def load_script(self):
        
        root = get_project_root()

        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)

        self.out_dir = config["out_directory"]

        self.out_dir = os.path.join(root, self.out_dir)

        date = datetime.datetime.now()
        date_string = date.strftime("%G%m%d")

        self.out_dir = os.path.join(self.out_dir, date_string)

        check_folder_exists(self.out_dir)

        # Create and configure logger
        logging.basicConfig(filename=self.out_dir+ "/" + "logging.log",
                            format='%(asctime)s %(message)s',
                            filemode='w')
        
        # Creating an object
        self.logger = logging.getLogger()

    def return_logger(self):
        return self.logger
    
    def return_config(self):
        return self.config
    
    def return_out_dir(self):
        return self.out_dir
    


