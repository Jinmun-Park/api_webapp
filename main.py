from src.db.chart import chart_export

#log = log_helper.get_logger(__name__)

def main():
   # Use a breakpoint in the code line below to debug your script.
   print("Popular Chart Export to postgresSQL")
   #log.info('Loading Database details')
   chart_export(key='update')

if __name__ == '__main__':
   #log_helper.log_setup()
   main()
