from src.utils.db2_helper import load_database_details
from src.models.gts_irm_hw_element import run_irm_hw_elements
import src.utils.log_helper as log_helper

log = log_helper.get_logger(__name__)


def main():
   # Use a breakpoint in the code line below to debug your script.
   print("GTS IRM HW Started...")
   log.info('Loading Database details')
   load_database_details()
   log.info('Running Model')
   run_irm_hw_elements()
   log.info('Completed All Run')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   log_helper.log_setup()
   main()
