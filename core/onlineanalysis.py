import numpy as np
import datetime

class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    DARK_GRAY = "\033[1;30m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLACK = "\033[0;30m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

def set_char(value, col):
    if value < chunk:
        c = str(value)
    else:
        c = "\u2588"
    return col+c+color.END

def summary(dates, status, chunk, nchunks):
    today = datetime.datetime.today().strftime("%A %d %B %Y")
    today = color.BOLD+color.BLUE+today+color.END
    title = f"GIGATL database status on {today}" 
    header = " "*4+"".join([f"{k:<5}" for k in range(0, nchunks, chunk)])

    string = []
    string += [title, header]
    for k in range(13):
        line = f"{k+1:2}: "
        for l in range(nchunks):
            if status[k, 0, l] > 0:
                c = set_char(status[k, 0, l], color.DARK_GRAY)
            elif status[k, 1, l] > 0:
                c = set_char(status[k, 1, l], color.RED)
            else:
                c = set_char(status[k, 2, l], color.BLUE+color.BOLD)
            line += c
        string += [line]

    d0 = datetime.datetime(*(int(x) for x in dates[0].split("-")))
    l0 = " "*4
    l1 = " "*4
    leftsep = "\u258F"
    for k in range(0, nchunks-chunk, 2*chunk):
        d = d0+(k*chunk)*datetime.timedelta(1)
        day = d.strftime("%d:%b:%y")
        l0 += f"{leftsep:<10}"
        l1 += f"{day:<10}"
    
    string += [l0, l1, ""]

    return "\n".join(string)

def analyze_filestatus(ds):
    dates = ds.dates
    ndates = len(dates)
    chunk = 5
    nchunks = int(np.ceil(ndates/chunk))
    status = np.zeros((13, 3, chunk*nchunks), dtype="b")

    for s in range(13): 
        r = ds.readers[s+1] 
        for k,d in enumerate(dates): 
            if d not in r.dates: 
                status[s,0,k] = 1 
            elif r.filestatus[d] == "lost": 
                status[s,1,k] = 1 
            elif r.filestatus[d] == "online": 
                status[s,2,k] = 1 


    status.shape = (13, 3, nchunks, chunk)
    status = status.sum(axis=-1)

    return (dates, status, chunk, nchunks)

if __name__ == "__main__":
    import bindatasets as bd

    ds = bd.Dataset()

    dates, status, chunk, nchunks = analyze_filestatus(ds)
    summ = summary(dates, status, chunk, nchunks)
    print(summ)

    # with open("status.txt", "w") as fid:
    #     fid.write(summ)
    
