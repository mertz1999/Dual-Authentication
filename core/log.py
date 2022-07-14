import datetime

# Log file
class Log():
    """
        We use thid class to save log information for future use.
        finname is year-month.log and it has been generated automaticaly.
    """
    def __init__(self) -> None:
        now = datetime.datetime.now()
        filename = str(now.year)+"_"+str(now.month)+".log"
        self.f = open('./logs/'+filename, "a")
    
    def make_log(self, data: str):
        now = datetime.datetime.now()
        print(now, data)
        self.f.write(f"({now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}) ")
        self.f.write(data)
        self.f.write('\n')
        
        


X = Log()
# f.write("Now the file has more content!")
# f.close()