import time
def clock():
    hours = 0
    minutes = 0

    while True:
        for seconds in range(60):
            print(f"{hours:02}:{minutes:02}:{seconds:02}")
            time.sleep(1)

            if seconds == 59:
                minutes += 1
                if minutes == 60:
                    minutes = 0
                    hours += 1
                    if hours == 24:
                        hours = 0


clock()
