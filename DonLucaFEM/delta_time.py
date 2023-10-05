import datetime

def delta_time(f):
    def wrapper(*args, **kwargs):
        t_0 = datetime.datetime.now()
        result = f(*args, **kwargs)
        t_1 = datetime.datetime.now()
        dt = t_1-t_0
        print(f'The operation took {dt.total_seconds()} seconds')
        
        return result
    return wrapper